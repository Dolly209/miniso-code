"""One M1-parameter -> M3/M4 -> M5 -> M6/M7 computation for fit and inference.

Only canonical frames enter here. Restoration is NOT refitted during parameter
search: each target retains its own OOF/final upstream model and cutoff.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Callable, Mapping, Any
import json
from hashlib import sha256

import numpy as np
import pandas as pd

from factors.engine import calculate_demand_adjustments
from factors.frozen import fit_factor_transform
from first_order.engine import calculate_first_order
from placement.engine import plan_placement


BASE = "feature__first_order__D_demand_base"
PREFIX = "feature__adjustments__"


def prepare_context_features(context: Mapping, transform: Mapping) -> dict:
    platform = transform.get("social_platform_transform")
    frames = context["frames"]
    if platform is None or "social_history" not in frames or context.get("prepared_transform_id") == transform.get("transform_id"):
        return dict(context)
    from factors.engine import build_social_small_factors
    alias_rows = frames.get("social_aliases", pd.DataFrame(columns=["sku_code", "external_ip_key"]))
    aliases = {str(k): set(g.external_ip_key.astype(str)) for k, g in alias_rows.groupby("sku_code")}
    cfg = context["parameters"]["factors"]
    features = build_social_small_factors(features=frames["features"],
        media_history=frames["social_history"], sku_social_aliases=aliases,
        prelaunch_days=int(cfg["social_prelaunch_days"]), momentum_days=int(cfg["social_momentum_days"]),
        minimum_days=int(cfg["minimum_social_days"]),
        decision_cutoff_date=pd.Timestamp(context["cutoff"]).date(), platform_transform=platform)
    return {**context, "frames": {**frames, "features": features}, "prepared_transform_id": transform.get("transform_id")}


def _target_frames(context: Mapping, sku: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = context["frames"]
    features = frames["features"]
    products = frames.get("products", features)
    targets = products.loc[products.sku_code.astype(str).eq(sku)].copy()
    if len(targets) != 1:
        raise ValueError(f"expected one canonical target product row: {sku}")
    summary = frames["comparable_summary"].loc[frames["comparable_summary"].sku_code.astype(str).eq(sku)].copy()
    return targets, summary


def full_first_order(context: Mapping, comparables: pd.DataFrame, *,
                     frozen_transform: Mapping, learned_model=None,
                     placement_result=None, include_details=True,
                     audit_outputs: dict | None = None):
    """The formal first-order entry and training replay both call this chain."""
    context = prepare_context_features(context, frozen_transform)
    f, p, sku = context["frames"], context["parameters"], context["target_sku"]
    targets, summary = _target_frames(context, sku)
    current = comparables.loc[comparables.selected.eq(1)]
    summary["selected_candidate_count"] = len(current)
    if "historical_velocity" in current:
        summary["anchor_velocity"] = float(np.dot(current.alpha_weight, current.historical_velocity))
    if "path_code" in current:
        summary["selected_path"] = "+".join(dict.fromkeys(current.path_code.astype(str)))
        summary["path_reliability"] = sum(float(r.alpha_weight) * float(p["comparable"]["base_reliability"][r.path_code]) for r in current.itertuples(index=False))
    if placement_result is None:
        placement_result = plan_placement(
            targets=targets, comparables=comparables, stores=f["stores"],
            restored_store_weekly=f["restored_store_weekly"],
            restored_tier_weekly=f["restored_tier_weekly"], config=p["placement"],
            placement_policy=f.get("placement_policy"), placement_evidence=f.get("placement_evidence"),
            target_store_features=f.get("target_store_features"),
            prepared_cache=context.get("prepared_placement"))
    factor_config = deepcopy(p["factors"])
    competition = f.get("competition_rules")
    if learned_model is not None:
        for group, initial_weights in factor_config["small_factor_weights"].items():
            learned = {name.split("delta_factor__", 1)[1]: max(float(coefficient), 0.)
                for name, coefficient, g in zip(learned_model.feature_columns, learned_model.coefficients, learned_model.feature_groups)
                if g == group and "delta_factor__" in name}
            beta = (sum(learned.values()) if learned else sum(float(c)
                for name, c, g in zip(learned_model.feature_columns, learned_model.coefficients, learned_model.feature_groups)
                if g == group and name.endswith("delta_" + group)))
            factor_config["group_betas"][group] = beta
            if beta > 1e-12:
                if learned:
                    factor_config["small_factor_weights"][group] = {n: learned.get(n, 0.) / beta for n in initial_weights}
        factor_config["price_elasticity"] = sum(float(c) for c, g in zip(learned_model.coefficients, learned_model.feature_groups) if g == "PRICE")
        if competition is not None:
            competition = competition.copy()
            coefficient = sum(float(c) for c, g in zip(learned_model.coefficients, learned_model.feature_groups) if g == "COMPETITION")
            competition["coefficient"] = np.exp(min(coefficient, 0.))
        factor_config["coefficient_bounds"] = learned_model.coefficient_bounds
        factor_config["price_bounds"] = learned_model.price_bounds
        factor_config["demand_bounds"] = learned_model.demand_bounds
    factors = calculate_demand_adjustments(
        targets=targets, features=f["features"], comparables=comparables,
        comparable_summary=summary, config=factor_config,
        competition_rules=competition, frozen_transform=frozen_transform)
    if audit_outputs is not None:
        audit_outputs.update(factor_detail=factors.detail.copy(),
                             transformed_features=factors.transformed_features.copy(),
                             demand_adjustments=factors.adjustments.copy(),
                             effective_factor_config=deepcopy(factor_config))
    enriched = factors.adjustments.copy()
    reference = factors.transformed_features.set_index("sku_code")
    chosen = comparables.loc[comparables.selected.eq(1) & ~comparables.candidate_sku_code.astype(str).str.startswith("__STATIC__")].reset_index(drop=True)
    if len(chosen) > 3:
        raise ValueError("Step5 restored candidate envelope is Top3; cannot expand K beyond restored evidence")
    # Carry the scores, not only their preweighted differences. Learning omega
    # must renormalise available features independently for target and each peer.
    for group in p["factors"]["small_factor_weights"].values():
        for name in group:
            enriched[f"__target_score__{name}"] = reference.loc[sku, f"score__{name}"]
            for j in range(3):
                code = str(chosen.iloc[j].candidate_sku_code) if j < len(chosen) else None
                enriched[f"__peer{j}_score__{name}"] = reference.loc[code, f"score__{name}"] if code in reference.index else np.nan
    for j in range(3):
        enriched[f"__peer{j}_alpha"] = float(chosen.iloc[j].alpha_weight) if j < len(chosen) else 0.0
    enriched["__external_coefficient"] = enriched.C_ext
    # Training does not need to rebuild store quantities for every small-factor
    # gradient. Its baseline is D_sell; production calls M6/M7 once below.
    if learned_model is not None:
        from training.section84 import predict_demand_residual
        encoded = pd.concat([pd.DataFrame({BASE: [1.0]}),
            enriched.drop(columns="sku_code").add_prefix(PREFIX),
            enriched[[c for c in enriched if c.startswith("__")]]], axis=1)
        encoded[learned_model.base_demand_column] = 1.0
        learned_multiplier = predict_demand_residual(learned_model, encoded)
        if not np.allclose(factors.adjustments.C_demand, learned_multiplier, rtol=1e-10, atol=1e-10):
            raise AssertionError("training factor kernel differs from formal M5 formula")
    result = calculate_first_order(
        targets=targets, placement_plan=placement_result.store_plan,
        demand_curve=placement_result.demand_curve, adjustments=factors.adjustments,
        comparable_summary=summary, config=p["first_order"], factor_config=p["factors"],
        constraints=f.get("constraints"), arrival_plan=f.get("arrival_plan"), include_details=include_details,
        placement_summary=placement_result.summary)
    if learned_model is not None and learned_model.interval_log_quantiles:
        intervals = result.decisions[["sku_code", "D_demand_adjusted"]].copy()
        for quantile in ("p10", "p90"):
            intervals[quantile] = intervals.D_demand_adjusted * np.exp(float(learned_model.interval_log_quantiles[quantile]))
        # Reuse M3/M4/M5; attach only Train-OOF calibrated uncertainty. No
        # reliability score is fabricated from the interval width.
        result = calculate_first_order(
            targets=targets, placement_plan=placement_result.store_plan,
            demand_curve=placement_result.demand_curve, adjustments=factors.adjustments,
            comparable_summary=summary, config=p["first_order"], factor_config=p["factors"],
            constraints=f.get("constraints"), arrival_plan=f.get("arrival_plan"), prediction_intervals=intervals,
            placement_summary=placement_result.summary)
    return result, placement_result, enriched


class FullChainReplay:
    """Bounded, label-free recompute cache with fold-local frozen transforms."""
    def __init__(self, context_loader: Callable[[str], Mapping], *, key_column: str,
                 base_demand_column: str = BASE, cache_size: int = 4096, progress=None):
        self.load = context_loader
        self.key = key_column
        self.base = base_demand_column
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.progress = progress
        self.calls = self.hits = 0
        self.transform_cache = {}
        self.feature_cache = OrderedDict()
        self.placement_cache = OrderedDict()

    def validate_inputs(self, rows: pd.DataFrame, candidate_detail: pd.DataFrame) -> None:
        required = ["candidate_sku_code", "selected", "alpha_weight", "historical_velocity", "quality_score", "path_priority"]
        for _, row in rows.iterrows():
            key = str(row[self.key])
            context = self.load(key)
            if str(row["sku_code"]) != context["target_sku"]:
                raise ValueError(f"canonical target mismatch: {key}")
            if "decision_cutoff_time" in row and pd.Timestamp(row.decision_cutoff_time) != pd.Timestamp(context["cutoff"]):
                raise ValueError(f"canonical cutoff mismatch: {key}")
            actual = context["frames"]["comparables"]
            actual = actual.loc[actual.target_sku_code.astype(str).eq(context["target_sku"])]
            expected = candidate_detail.loc[candidate_detail[self.key].astype(str).eq(key)]
            try:
                # A PATH4-only bundle has no pairwise similarity dimensions.
                # Concatenating it with other groups adds all-null columns to
                # the summary CSV. Missing optional dimensions and explicit
                # nulls are equivalent, never a zero similarity. Core fields
                # are still required; any populated value on either side is
                # compared and cannot be hidden by taking an intersection.
                actual[required]
                expected[required]
                optional = sorted({c for f in (actual, expected) for c in f if c.startswith("similarity_")})
                fields = required + optional
                pd.testing.assert_frame_equal(actual.reindex(columns=fields).sort_values("candidate_sku_code").reset_index(drop=True),
                    expected.reindex(columns=fields).sort_values("candidate_sku_code").reset_index(drop=True),
                    check_dtype=False, rtol=1e-9, atol=1e-12)
            except (AssertionError, KeyError) as exc:
                raise ValueError(f"candidate detail and canonical recompute snapshot differ: {key}") from exc

    def fit_transform(self, fit_rows: pd.DataFrame) -> dict:
        if "split" in fit_rows and fit_rows.split.astype(str).str.lower().ne("train").any():
            raise ValueError("CDF fit contains non-Train rows")
        cache_key = tuple(sorted(fit_rows[self.key].astype(str)))
        if cache_key in self.transform_cache:
            return self.transform_cache[cache_key]
        pieces, keys, histories = [], [], []
        config = None
        for key in cache_key:
            context = self.load(key)
            if "social_history" in context["frames"]:
                histories.append(context["frames"]["social_history"])
        platform_transform = None
        if histories:
            from factors.frozen import fit_social_platform_transform
            history = pd.concat(histories, ignore_index=True).drop_duplicates(
                ["stat_date", "external_ip_key", "platform_code"])
            platform_transform = fit_social_platform_transform(history)
        social_transform = {"social_platform_transform": platform_transform, "transform_id": "fit-" + sha256(json.dumps(platform_transform, sort_keys=True).encode()).hexdigest()[:16]}
        for row in fit_rows.itertuples(index=False):
            key = str(getattr(row, self.key))
            context = self.load(key)
            context = prepare_context_features(context, social_transform)
            source = context["frames"]["features"]
            target = source.loc[source.sku_code.astype(str).eq(context["target_sku"])].copy()
            pieces.append(target)
            keys.append(key)
            config = context["parameters"]["factors"]
        transform = fit_factor_transform(pd.concat(pieces, ignore_index=True),
            factor_weights=config["small_factor_weights"], winsor=tuple(config["winsor_quantiles"]))
        transform["fit_checkpoint_keys"] = sorted(keys)
        transform["social_platform_transform"] = platform_transform
        transform["transform_id"] = sha256(json.dumps(transform, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
        self.transform_cache[cache_key] = transform
        return transform

    def recompute(self, rows: pd.DataFrame, model, transform: Mapping) -> pd.DataFrame:
        from training.comparable_tuning import select_comparable_parameters
        output = []
        for _, row in rows.iterrows():
            key = str(row[self.key])
            context = self.load(key)
            feature_key = (key, transform["transform_id"])
            placement_key = context.get("bundle_identity", key)
            if placement_key not in self.placement_cache:
                self.placement_cache[placement_key] = {}
                while len(self.placement_cache) > 32:
                    self.placement_cache.popitem(last=False)
            self.placement_cache.move_to_end(placement_key)
            context = {**context, "prepared_placement": self.placement_cache[placement_key]}
            if feature_key not in self.feature_cache:
                prepared = prepare_context_features(context, transform)
                self.feature_cache[feature_key] = prepared["frames"]["features"]
                while len(self.feature_cache) > 256:
                    self.feature_cache.popitem(last=False)
            self.feature_cache.move_to_end(feature_key)
            context = {**context, "frames": {**context["frames"], "features": self.feature_cache[feature_key]},
                       "prepared_transform_id": transform["transform_id"]}
            detail = context["frames"]["comparables"]
            detail = detail.loc[detail.target_sku_code.astype(str).eq(context["target_sku"])].copy()
            selected = select_comparable_parameters(detail, model)
            signature = tuple((str(r.candidate_sku_code), float(r.alpha_weight)) for r in selected.itertuples(index=False))
            token = (key, signature, transform["transform_id"])
            self.calls += 1
            if token not in self.cache:
                result, placement, factors = full_first_order(context, selected, frozen_transform=transform, include_details=False)
                decision = result.decisions.iloc[0].to_dict()
                prior_cover = row.get("feature__first_order__cover_weeks")
                if pd.notna(prior_cover) and int(prior_cover) != int(decision["cover_weeks"]):
                    raise ValueError(f"W_cover changed between modeling row and recompute inputs: {key}")
                numeric = {self.base: float(decision["D_demand_base"]),
                           "step5_selected_store_count": int(placement.store_plan.selected.sum()),
                           "step5_selected_candidates": "|".join(selected.candidate_sku_code.astype(str)),
                           "step5_alpha_weights": json.dumps(selected.alpha_weight.tolist()),
                           "step5_transform_id": transform["transform_id"]}
                numeric["step5_cover_weeks"] = int(decision["cover_weeks"])
                numeric.update({"feature__first_order__" + name: value
                    for name, value in decision.items() if name != "sku_code"})
                for name, value in factors.iloc[0].items():
                    if name != "sku_code":
                        numeric[PREFIX + name] = value
                    if name.startswith("__"):
                        numeric[name] = value
                self.cache[token] = numeric
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            else:
                self.hits += 1
            self.cache.move_to_end(token)
            output.append({**row.to_dict(), **self.cache[token]})
            if self.progress and (self.calls == 1 or self.calls % 25 == 0):
                self.progress({"evaluations": self.calls, "cache_hits": self.hits, "checkpoint": key})
        return pd.DataFrame(output)

    def infer_orders(self, rows: pd.DataFrame, comparable_model, demand_model) -> pd.DataFrame:
        from training.comparable_tuning import select_comparable_parameters
        output = []
        for _, row in rows.iterrows():
            context = self.load(str(row[self.key]))
            detail = context["frames"]["comparables"]
            chosen = select_comparable_parameters(detail.loc[
                detail.target_sku_code.astype(str).eq(context["target_sku"])], comparable_model)
            result, _, _ = full_first_order(context, chosen,
                frozen_transform=demand_model.frozen_transform, learned_model=demand_model)
            output.append({self.key: row[self.key], **result.decisions.iloc[0].to_dict()})
        return pd.DataFrame(output)


def compile_group_scores(frame: pd.DataFrame, feature_names: list[str]) -> tuple:
    prefixes = ["__target_score__"] + [f"__peer{j}_score__" for j in range(3)]
    arrays = [frame[[prefix + n for n in feature_names]].to_numpy(dtype=float) for prefix in prefixes]
    return (*arrays, frame[[f"__peer{j}_alpha" for j in range(3)]].to_numpy(float))


def exact_group_effect(frame: pd.DataFrame, feature_names: list[str], coefficients: np.ndarray,
                       compiled: tuple | None = None) -> np.ndarray:
    """beta=sum(theta), omega=theta/beta, separate availability renormalisation."""
    beta = float(np.maximum(coefficients, 0).sum())
    if beta <= 1e-12:
        return np.zeros(len(frame))
    w = np.maximum(coefficients, 0) / beta
    arrays = compiled if compiled is not None else compile_group_scores(frame, feature_names)
    def score(values):
        denom = np.isfinite(values) @ w
        return np.divide(np.nan_to_num(values) @ w, denom,
                         out=np.full(len(frame), np.nan), where=denom > 0)
    target = score(arrays[0])
    peers = np.column_stack([score(values) for values in arrays[1:4]])
    alpha = arrays[4]
    alpha = np.where(np.isfinite(peers), alpha, 0.0)
    denom = alpha.sum(axis=1)
    reference = np.divide((np.nan_to_num(peers) * alpha).sum(axis=1), denom,
                          out=np.full(len(frame), np.nan), where=denom > 0)
    return beta * np.nan_to_num((target - reference) / 100.0)
