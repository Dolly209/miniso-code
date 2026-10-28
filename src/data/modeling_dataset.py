from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from data.artifacts import StageArtifactStore
from data.domain import require_columns
from data.forecast_eligibility import forecast_fit_population


FEATURE_CUTOFF_COLUMNS = ("feature_cutoff_time", "decision_cutoff_time")
FIRST_ORDER_LABEL_COLUMN = "D_first_ref"


def assess_modeling_readiness(
    *,
    cohort: pd.DataFrame,
    feature_frames: Sequence[tuple[str, pd.DataFrame]],
    label_frame: pd.DataFrame | None = None,
    label_column: str = FIRST_ORDER_LABEL_COLUMN,
    required_splits: Sequence[str] = ("train", "test"),
) -> dict[str, object]:
    """Audit split coverage, mature labels, and historical feature cutoffs."""

    require_columns(cohort, ("sku_code", "first_actual_sale_date", "split"), "cohort")
    checkpoint_mode = "prediction_checkpoint_key" in cohort
    key_column = "prediction_checkpoint_key" if checkpoint_mode else "sku_code"
    base_columns = ["sku_code", "first_actual_sale_date", "split"]
    if "dataset_version" in cohort:
        base_columns.append("dataset_version")
    if checkpoint_mode:
        require_columns(
            cohort,
            ("prediction_checkpoint_key", "decision_cutoff_time", "prediction_checkpoint"),
            "prelaunch checkpoint cohort",
        )
        base_columns.extend(
            ["prediction_checkpoint_key", "decision_cutoff_time", "prediction_checkpoint"]
        )
    base = cohort[base_columns].copy()
    base["sku_code"] = base["sku_code"].astype(str)
    if base[key_column].astype(str).duplicated().any():
        raise ValueError(f"cohort is not unique by {key_column}")
    base["first_actual_sale_date"] = pd.to_datetime(
        base["first_actual_sale_date"], errors="coerce"
    )
    if base["first_actual_sale_date"].isna().any():
        raise ValueError("cohort contains invalid first_actual_sale_date")
    split = base["split"].astype(str).str.lower()
    split_counts = {
        "Train": int(base.loc[split.eq("train"), "sku_code"].nunique()),
        "Test": int(base.loc[split.eq("test"), "sku_code"].nunique()),
    }
    if label_frame is None:
        raise ValueError(
            "formal first-order modeling requires an external W_cover label file; "
            "sales_26w_qty in the cohort is not D_first_ref"
        )
    if label_column == "sales_26w_qty":
        raise ValueError(
            "sales_26w_qty is a 26-week maturity/distribution field, not the "
            "V3.2 first-order label; use D_first_ref from build_first_order_labels.py"
        )
    labels = label_frame
    require_columns(labels, ("sku_code", label_column), "training labels")
    label_rows = labels[["sku_code", label_column]].copy()
    label_rows["sku_code"] = label_rows["sku_code"].astype(str)
    if label_rows["sku_code"].duplicated().any():
        conflicts = label_rows.groupby("sku_code")[label_column].nunique(dropna=False)
        if conflicts.gt(1).any():
            raise ValueError("training labels conflict within sku_code")
        label_rows = label_rows.drop_duplicates("sku_code")
    label_rows[label_column] = pd.to_numeric(label_rows[label_column], errors="coerce")
    label_joined = base.merge(
        label_rows,
        on="sku_code",
        how="left",
        validate="many_to_one" if checkpoint_mode else "one_to_one",
    )
    mature = label_joined[label_column].notna()
    label_split = label_joined["split"].astype(str).str.lower()
    mature_split_counts = {
        "Train": int(label_joined.loc[mature & label_split.eq("train"), "sku_code"].nunique()),
        "Test": int(label_joined.loc[mature & label_split.eq("test"), "sku_code"].nunique()),
    }
    feature_reports: dict[str, object] = {}
    blocking: list[str] = []
    required = {str(value).strip().lower() for value in required_splits}
    if not required or not required.issubset({"train", "test"}):
        raise ValueError("required_splits must contain train and/or test")
    if "train" in required and split_counts["Train"] == 0:
        blocking.append("no Train SKUs in selected cohort")
    if "test" in required and split_counts["Test"] == 0:
        blocking.append("no Test SKUs in selected cohort")
    if "train" in required and mature_split_counts["Train"] == 0:
        blocking.append("no mature non-null Train labels")
    if "test" in required and mature_split_counts["Test"] == 0:
        blocking.append("no mature non-null Test labels")
    if not feature_frames:
        blocking.append("no feature artifacts supplied")
    for namespace, frame in feature_frames:
        require_columns(frame, ("sku_code",), f"feature frame {namespace}")
        source = frame.copy()
        source["sku_code"] = source["sku_code"].astype(str)
        if checkpoint_mode and key_column not in source:
            blocking.append(
                f"feature artifact {namespace} is SKU-level, not keyed by prediction_checkpoint_key"
            )
            join_column = "sku_code"
        else:
            join_column = key_column
        if source[join_column].astype(str).duplicated().any():
            raise ValueError(
                f"feature frame {namespace} is not unique by {join_column}"
            )
        cutoff_column = next(
            (column for column in FEATURE_CUTOFF_COLUMNS if column in source), None
        )
        joined = base.merge(
            source[[join_column]], on=join_column, how="left", indicator=True,
            validate="many_to_one" if checkpoint_mode and join_column == "sku_code" else "one_to_one",
        )
        coverage = joined["_merge"].eq("both")
        covered_skus = joined.loc[coverage, "sku_code"].nunique()
        namespace_report: dict[str, object] = {
            "rows": len(source),
            "covered_cohort_rows": int(coverage.sum()),
            "covered_cohort_skus": int(covered_skus),
            "coverage_rate": float(coverage.mean()) if len(coverage) else 0.0,
            "feature_cutoff_column": cutoff_column,
            "point_in_time_verified_rows": 0,
            "future_leakage_rows": 0,
            "missing_feature_cutoff_rows": 0,
        }
        if namespace == "first_order" and "training_feature_eligible" in source:
            unusable = source["training_feature_eligible"].astype(str).str.lower().isin(("false", "0"))
            namespace_report["blocked_forecast_rows"] = int(unusable.sum())
            namespace_report["blocked_forecast_skus"] = source.loc[unusable, "sku_code"].astype(str).tolist()
            if unusable.any():
                try:
                    population = forecast_fit_population(base, source)
                    omitted = population.loc[~population["forecast_available"]]
                    namespace_report["fit_excluded_forecasts"] = omitted.to_dict(orient="records")
                    usable_train = population["forecast_available"] & population["split"].astype(str).str.lower().eq("train")
                    namespace_report["usable_train_skus"] = int(population.loc[usable_train, "sku_code"].nunique())
                    if "train" in required and not usable_train.any():
                        blocking.append("first_order unavailable forecasts leave no usable Train samples")
                except ValueError as exc:
                    blocking.append(str(exc))
        if int(coverage.sum()) == 0:
            blocking.append(f"feature artifact {namespace} covers no cohort SKUs")
        elif int(coverage.sum()) != len(base):
            blocking.append(
                f"feature artifact {namespace} misses "
                f"{int((~coverage).sum())} selected cohort rows"
            )
        if cutoff_column is None:
            blocking.append(
                f"feature artifact {namespace} has no feature_cutoff_time/decision_cutoff_time"
            )
        else:
            cutoff = source[[join_column, cutoff_column]].copy().rename(
                columns={cutoff_column: "__feature_cutoff_time"}
            )
            # Compare instants rather than wall-clock strings.  Historical
            # checkpoint artifacts normally carry +08:00 while some adapter
            # outputs are UTC; both represent the same cutoff after this
            # normalization.
            cutoff["__feature_cutoff_time"] = pd.to_datetime(
                cutoff["__feature_cutoff_time"], errors="coerce", utc=True
            )
            checked = base.merge(
                cutoff,
                on=join_column,
                how="inner",
                validate="many_to_one" if checkpoint_mode and join_column == "sku_code" else "one_to_one",
            )
            feature_cutoff = checked["__feature_cutoff_time"]
            missing_cutoff = feature_cutoff.isna()
            if checkpoint_mode:
                scheduled_cutoff = pd.to_datetime(
                    checked["decision_cutoff_time"], errors="coerce", utc=True
                )
                future = feature_cutoff.gt(scheduled_cutoff)
                cutoff_mismatch = (
                    feature_cutoff.notna()
                    & scheduled_cutoff.notna()
                    & feature_cutoff.ne(scheduled_cutoff)
                )
                namespace_report["checkpoint_cutoff_mismatch_rows"] = int(
                    cutoff_mismatch.sum()
                )
                if cutoff_mismatch.any():
                    blocking.append(
                        f"feature artifact {namespace} has {int(cutoff_mismatch.sum())} rows not frozen at their scheduled checkpoint"
                    )
            else:
                first_sale_utc = pd.to_datetime(
                    checked["first_actual_sale_date"], errors="coerce", utc=True
                )
                future = feature_cutoff.dt.date.gt(first_sale_utc.dt.date)
            namespace_report["point_in_time_verified_rows"] = int(
                (~missing_cutoff & ~future).sum()
            )
            namespace_report["future_leakage_rows"] = int(future.sum())
            namespace_report["missing_feature_cutoff_rows"] = int(missing_cutoff.sum())
            if missing_cutoff.any():
                blocking.append(
                    f"feature artifact {namespace} has {int(missing_cutoff.sum())} missing cutoffs"
                )
            if future.any():
                blocking.append(
                    f"feature artifact {namespace} has {int(future.sum())} rows after first sale"
                )
        feature_reports[namespace] = namespace_report
    return {
        "status": "ready_for_train_oof_and_locked_test" if not blocking else "blocked",
        "split_sku_counts": split_counts,
        "mature_label_rows": int(mature.sum()),
        "mature_label_sku_counts": mature_split_counts,
        "missing_label_rows": int((~mature).sum()),
        "feature_artifacts": feature_reports,
        "point_in_time_policy": (
            "each M5/M4/M3 prelaunch forecast is an independent row keyed by "
            "prediction_checkpoint_key; every feature cutoff must equal its scheduled "
            "release-minus-month checkpoint and precede first sale"
            if checkpoint_mode
            else "every historical feature row must carry its own cutoff and the cutoff must not be later than that SKU's first actual sale date"
        ),
        "row_key": key_column,
        "independent_checkpoint_rows": checkpoint_mode,
        "blocking_reasons": blocking,
        "required_splits": sorted(required),
    }


def assemble_modeling_dataset(
    *,
    cohort: pd.DataFrame,
    feature_frames: Sequence[tuple[str, pd.DataFrame]],
    label_frame: pd.DataFrame | None = None,
    label_column: str = FIRST_ORDER_LABEL_COLUMN,
) -> pd.DataFrame:
    """Join standardized SKU or independent prelaunch-checkpoint artifacts."""

    require_columns(cohort, ("sku_code", "first_actual_sale_date", "split"), "cohort")
    checkpoint_mode = "prediction_checkpoint_key" in cohort
    key_column = "prediction_checkpoint_key" if checkpoint_mode else "sku_code"
    base_columns = ["sku_code", "first_actual_sale_date", "split"]
    if "dataset_version" in cohort:
        base_columns.append("dataset_version")
    if checkpoint_mode:
        base_columns.extend(
            [
                "prediction_checkpoint_key",
                "prediction_checkpoint",
                "lead_months",
                "decision_cutoff_time",
                "decision_date",
                "release_dt",
                "forecast_scope",
                "checkpoint_weight_policy",
            ]
        )
    for optional in (
        "time_block",
        "pilot_version",
        "evaluation_scope",
        "label_source_model_health",
        "label_source_model_id",
        "sales_26w_qty",
        "long_tail_weight",
        "label_definition",
        label_column,
    ):
        if optional in cohort and optional not in base_columns:
            base_columns.append(optional)
    result = cohort[base_columns].copy()
    if label_frame is None:
        raise ValueError(
            "formal first-order modeling requires an external W_cover label file"
        )
    if label_column == "sales_26w_qty":
        raise ValueError(
            "sales_26w_qty cannot be used as the V3.2 first-order label"
        )
    labels = label_frame
    require_columns(labels, ("sku_code", label_column), "training labels")
    if label_column not in result:
        label_rows = labels[["sku_code", label_column]].copy()
        if label_rows["sku_code"].duplicated().any():
            conflicts = label_rows.groupby("sku_code")[label_column].nunique(dropna=False)
            if conflicts.gt(1).any():
                raise ValueError("training labels conflict within sku_code")
            label_rows = label_rows.drop_duplicates("sku_code")
        result = result.merge(
            label_rows,
            on="sku_code",
            how="left",
            validate="many_to_one" if checkpoint_mode else "one_to_one",
        )
    result = result.rename(columns={label_column: "label"})
    for namespace, frame in feature_frames:
        require_columns(frame, ("sku_code",), f"feature frame {namespace}")
        if checkpoint_mode and key_column not in frame:
            raise ValueError(
                f"feature frame {namespace} must be keyed by prediction_checkpoint_key; "
                "replicating one current SKU snapshot across M5/M4/M3 would leak time"
            )
        join_columns = ["sku_code", key_column] if checkpoint_mode else ["sku_code"]
        join_columns = list(dict.fromkeys(join_columns))
        if frame[key_column].astype(str).duplicated().any():
            raise ValueError(f"feature frame {namespace} is not unique by {key_column}")
        columns = [name for name in frame.columns if name not in join_columns]
        selected = frame[[*join_columns, *columns]].copy()
        selected = selected.rename(
            columns={name: f"feature__{namespace}__{name}" for name in columns}
        )
        result = result.merge(
            selected,
            on=join_columns,
            how="left",
            validate="one_to_one",
        )
    result["label"] = pd.to_numeric(result["label"], errors="coerce")
    result["label_mature"] = result["label"].notna()
    population = forecast_fit_population(result, result, prefix="feature__first_order__")
    result["forecast_feature_available"] = population["forecast_available"].to_numpy()
    result["parameter_fit_eligible"] = (
        population["forecast_available"] & population["split"].astype(str).str.lower().eq("train")
    ).to_numpy()
    result["parameter_fit_exclusion_reason"] = population["fit_exclusion_reason"].to_numpy()
    return result.loc[result["label_mature"]].reset_index(drop=True)


def read_feature_artifacts(paths: Sequence[tuple[str, Path]]) -> list[tuple[str, pd.DataFrame]]:
    store = StageArtifactStore()
    return [(namespace, store.read_frame(path)) for namespace, path in paths]
