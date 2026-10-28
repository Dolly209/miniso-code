from __future__ import annotations

from datetime import date, datetime, timedelta
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from data.domain import normalize_sku_column, require_columns
from data.domestic import DataContractViolation
from data.launch_contract import validate_launch_origin_contract, validate_launch_universe_pair
from training.weights import long_tail_weights


REPLAY_SCHEDULE_COLUMNS = (
    "prediction_checkpoint_key",
    "prediction_checkpoint",
    "sku_code",
    "decision_cutoff_time",
    "first_actual_sale_date",
    "release_dt",
    "split",
    "dataset_version",
)

CANDIDATE_IDENTITY_COLUMNS = (
    "dataset_version",
    "first_actual_sale_date",
    "split",
    "sales_26w_qty",
    "long_tail_weight",
)

CHECKPOINT_IDENTITY_COLUMNS = (
    "prediction_checkpoint_key",
    "prediction_checkpoint",
    "lead_months",
    "decision_cutoff_time",
    "decision_date",
    "release_dt",
    "first_actual_sale_date",
    "split",
)


def _parse_aware(value: object, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataContractViolation(f"{field} contains an invalid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DataContractViolation(f"{field} must include a UTC offset")
    return parsed


def normalize_replay_schedule(
    schedule: pd.DataFrame,
    *,
    available_through: datetime,
    sku_codes: Sequence[str] = (),
    limit_skus: int | None = None,
) -> pd.DataFrame:
    """Validate and select complete M5/M4/M3 SKU checkpoint sets."""

    require_columns(schedule, REPLAY_SCHEDULE_COLUMNS, "prelaunch replay schedule")
    if available_through.tzinfo is None or available_through.utcoffset() is None:
        raise ValueError("available_through must be timezone-aware")
    frame = normalize_sku_column(schedule)
    validate_launch_origin_contract(frame, name="prelaunch replay schedule")
    if frame["prediction_checkpoint_key"].astype(str).duplicated().any():
        raise DataContractViolation(
            "prelaunch replay schedule is not unique by prediction_checkpoint_key"
        )
    frame["first_actual_sale_date"] = pd.to_datetime(
        frame["first_actual_sale_date"], errors="coerce"
    )
    frame["release_dt"] = pd.to_datetime(frame["release_dt"], errors="coerce")
    cutoffs = pd.to_datetime(
        frame["decision_cutoff_time"], errors="coerce", utc=True
    )
    if (
        frame["first_actual_sale_date"].isna().any()
        or frame["release_dt"].isna().any()
        or cutoffs.isna().any()
    ):
        raise DataContractViolation(
            "prelaunch replay schedule contains invalid first-sale/release/cutoff values"
        )
    available_utc = pd.Timestamp(available_through).tz_convert("UTC")
    if cutoffs.gt(available_utc).any():
        examples = frame.loc[
            cutoffs.gt(available_utc),
            ["prediction_checkpoint_key", "decision_cutoff_time"],
        ].head(5)
        raise DataContractViolation(
            "checkpoint cutoff is after the global available-through time: "
            f"{examples.to_dict(orient='records')}"
        )
    first_sale_utc = pd.to_datetime(
        frame["first_actual_sale_date"], errors="coerce", utc=True
    )
    if cutoffs.dt.date.ge(first_sale_utc.dt.date).any():
        examples = frame.loc[
            cutoffs.dt.date.ge(first_sale_utc.dt.date),
            ["prediction_checkpoint_key", "decision_cutoff_time", "first_actual_sale_date"],
        ].head(5)
        raise DataContractViolation(
            "prelaunch checkpoint is not strictly before first sale: "
            f"{examples.to_dict(orient='records')}"
        )
    requested = list(dict.fromkeys(str(value).strip() for value in sku_codes))
    if any(not value for value in requested):
        raise ValueError("sku_codes must contain non-empty values")
    if requested:
        missing = sorted(set(requested) - set(frame["sku_code"]))
        if missing:
            raise DataContractViolation(
                f"requested replay SKUs are absent from schedule: {missing}"
            )
        keep = set(requested)
    elif limit_skus is not None:
        if limit_skus <= 0:
            raise ValueError("limit_skus must be positive")
        ordered = (
            frame[["sku_code", "first_actual_sale_date"]]
            .drop_duplicates("sku_code")
            .sort_values(
                ["first_actual_sale_date", "sku_code"],
                ascending=[False, True],
                kind="stable",
            )
        )
        keep = set(ordered.head(limit_skus)["sku_code"])
    else:
        keep = set(frame["sku_code"])
    selected = frame.loc[frame["sku_code"].isin(keep)].copy()
    checkpoint_sets = selected.groupby("sku_code")["prediction_checkpoint"].agg(
        lambda values: tuple(sorted(set(str(value) for value in values)))
    )
    # A new M3-only dataset is a complete schedule. For a multi-checkpoint
    # dataset every selected SKU must still have the same declared global set.
    expected = tuple(sorted(set(frame["prediction_checkpoint"].astype(str))))
    if not expected or not set(expected).issubset({"M3", "M4", "M5"}):
        raise DataContractViolation("prelaunch replay schedule contains unsupported checkpoints")
    # Series.ne(tuple) is interpreted as an element-wise vector comparison by
    # some pandas versions.  Compare each grouped tuple explicitly so the
    # validation is stable on the Python/pandas version used by the server.
    incomplete = checkpoint_sets.loc[
        checkpoint_sets.map(lambda value: value != expected)
    ]
    if not incomplete.empty:
        raise DataContractViolation(
            "selected SKUs do not all contain complete configured checkpoints "
            f"{expected}: "
            f"{incomplete.head(5).to_dict()}"
        )
    duplicate_sku_cutoff = selected.assign(__cutoff=cutoffs.loc[selected.index]).duplicated(
        ["sku_code", "__cutoff"]
    )
    if duplicate_sku_cutoff.any():
        raise DataContractViolation(
            "one SKU has multiple checkpoints at the same instant"
        )
    selected["__cutoff_utc"] = cutoffs.loc[selected.index]
    return selected.sort_values(
        ["__cutoff_utc", "sku_code", "prediction_checkpoint"], kind="stable"
    ).reset_index(drop=True)


def collapse_candidate_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Collapse checkpoint rows only when candidate-relevant attributes agree."""

    require_columns(
        cohort,
        ("sku_code", *CANDIDATE_IDENTITY_COLUMNS),
        "historical candidate cohort",
    )
    frame = normalize_sku_column(cohort)
    validate_launch_origin_contract(frame, name="historical candidate cohort")
    if frame["sku_code"].duplicated().any():
        conflicts = frame.groupby("sku_code", dropna=False)[
            list(CANDIDATE_IDENTITY_COLUMNS)
        ].nunique(dropna=False)
        bad = conflicts.gt(1).any(axis=1)
        if bad.any():
            raise DataContractViolation(
                "historical candidate cohort has conflicting duplicate SKU rows: "
                f"{bad.loc[bad].head(5).index.astype(str).tolist()}"
            )
        frame = frame.drop_duplicates("sku_code", keep="first")
    frame["first_actual_sale_date"] = pd.to_datetime(
        frame["first_actual_sale_date"], errors="coerce"
    )
    if frame["first_actual_sale_date"].isna().any():
        raise DataContractViolation(
            "historical candidate cohort contains invalid first_actual_sale_date"
        )
    versions = {
        str(value).strip()
        for value in frame["dataset_version"]
        if pd.notna(value) and str(value).strip()
    }
    if len(versions) != 1:
        raise DataContractViolation(
            "historical candidate cohort must contain exactly one dataset_version"
        )
    return frame.reset_index(drop=True)


def align_candidate_cohort_to_schedule(
    cohort: pd.DataFrame,
    schedule: pd.DataFrame,
) -> tuple[pd.DataFrame, date]:
    """Refit candidate Train/Test roles and long-tail weights to final labels.

    Round 3 artifacts can retain an older illustrative split. Once W_cover
    eligibility has been established, V3.2 requires the fixed split and
    long-tail weights to be refit on that final feasible cohort. This helper
    applies the final schedule boundary to the wider historical candidate
    universe and never lets a final-Test SKU alter the fit weights.
    """

    validate_launch_universe_pair(cohort, schedule)
    require_columns(
        schedule,
        ("sku_code", "dataset_version", "first_actual_sale_date", "split"),
        "prelaunch replay schedule",
    )
    schedule_frame = normalize_sku_column(schedule)
    schedule_frame["first_actual_sale_date"] = pd.to_datetime(
        schedule_frame["first_actual_sale_date"], errors="coerce"
    )
    if schedule_frame["first_actual_sale_date"].isna().any():
        raise DataContractViolation(
            "prelaunch replay schedule contains invalid first_actual_sale_date"
        )
    versions = {
        str(value).strip()
        for value in schedule_frame["dataset_version"]
        if pd.notna(value) and str(value).strip()
    }
    if len(versions) != 1:
        raise DataContractViolation(
            "prelaunch replay schedule must contain exactly one dataset_version"
        )
    target_version = next(iter(versions))
    if "temporal_split_boundary_date" in schedule_frame:
        parsed_boundaries = pd.to_datetime(
            schedule_frame["temporal_split_boundary_date"], errors="coerce"
        ).dropna().dt.date.unique()
        if len(parsed_boundaries) != 1:
            raise DataContractViolation(
                "prelaunch replay schedule must contain one temporal split boundary"
            )
        boundary = parsed_boundaries[0]
    else:
        train_dates = schedule_frame.loc[
            schedule_frame["split"].astype(str).str.lower().eq("train"),
            "first_actual_sale_date",
        ]
        if train_dates.empty:
            raise DataContractViolation(
                "prelaunch replay schedule contains no Train SKU"
            )
        boundary = train_dates.max().date()
    expected_train = schedule_frame["first_actual_sale_date"].dt.date.le(boundary)
    actual_train = schedule_frame["split"].astype(str).str.lower().eq("train")
    if expected_train.ne(actual_train).any():
        raise DataContractViolation(
            "prelaunch replay schedule split is inconsistent with its temporal boundary"
        )

    candidates = collapse_candidate_cohort(cohort)
    candidates["sales_26w_qty"] = pd.to_numeric(
        candidates["sales_26w_qty"], errors="coerce"
    )
    if (
        candidates["sales_26w_qty"].isna().any()
        or candidates["sales_26w_qty"].lt(0).any()
    ):
        raise DataContractViolation(
            "historical candidate cohort has invalid sales_26w_qty"
        )
    candidates["source_dataset_version"] = candidates["dataset_version"].astype(str)
    candidates["source_split"] = candidates["split"].astype(str)
    candidates["source_long_tail_weight"] = pd.to_numeric(
        candidates["long_tail_weight"], errors="coerce"
    )
    candidates["dataset_version"] = target_version
    candidates["split"] = pd.Series(
        candidates["first_actual_sale_date"].dt.date.le(boundary),
        index=candidates.index,
    ).map({True: "Train", False: "Test"})
    train = candidates.loc[candidates["split"].eq("Train")]
    if train.empty:
        raise DataContractViolation(
            "final temporal boundary leaves no Train historical candidates"
        )
    labels = dict(
        zip(
            train["sku_code"].astype(str),
            train["sales_26w_qty"].astype(float),
        )
    )
    weights = {
        row.sku_code: row.weight for row in long_tail_weights(labels)
    }
    candidates["long_tail_weight"] = (
        candidates["sku_code"].map(weights).fillna(1.0).astype(float)
    )
    return candidates.reset_index(drop=True), boundary


def candidate_pool_at_cutoff(
    cohort: pd.DataFrame,
    *,
    cutoff_time: datetime,
    minimum_mature_weeks: int,
    candidate_limit: int | None = None,
) -> pd.DataFrame:
    """Select only SKUs with the required observed lifecycle weeks by cutoff."""

    if cutoff_time.tzinfo is None or cutoff_time.utcoffset() is None:
        raise ValueError("cutoff_time must be timezone-aware")
    if minimum_mature_weeks <= 0:
        raise ValueError("minimum_mature_weeks must be positive")
    frame = collapse_candidate_cohort(cohort)
    latest_launch = cutoff_time.date() - timedelta(
        days=minimum_mature_weeks * 7 - 1
    )
    selected = frame.loc[
        frame["first_actual_sale_date"].dt.date.le(latest_launch)
    ].copy()
    selected = selected.sort_values(
        ["first_actual_sale_date", "sku_code"],
        ascending=[False, True],
        kind="stable",
    )
    if candidate_limit is not None:
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        # The cap is a smoke-only execution aid.  Prefer Train rows so a small
        # smoke run actually exercises M2 fitting instead of accidentally
        # selecting only the chronologically latest locked-Test rows.  Formal
        # replay omits this cap and therefore keeps the complete historical
        # candidate universe in its natural time order.
        train = selected.loc[
            selected["split"].astype(str).str.lower().eq("train")
        ].head(candidate_limit)
        remainder = selected.loc[~selected.index.isin(train.index)].head(
            max(candidate_limit - len(train), 0)
        )
        selected = pd.concat([train, remainder], ignore_index=False).copy()
    selected["first_actual_sale_date"] = selected[
        "first_actual_sale_date"
    ].dt.date.astype(str)
    return selected.reset_index(drop=True)


def attach_checkpoint_identity(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the unique scheduled checkpoint to one-row-per-target outputs."""

    require_columns(frame, ("sku_code",), "checkpoint feature output")
    identity = [
        column for column in CHECKPOINT_IDENTITY_COLUMNS if column in targets
    ]
    require_columns(
        targets,
        ("sku_code", "prediction_checkpoint_key", "decision_cutoff_time"),
        "checkpoint target rows",
    )
    keys = targets[["sku_code", *[value for value in identity if value != "sku_code"]]].copy()
    keys["sku_code"] = keys["sku_code"].astype(str)
    if keys["sku_code"].duplicated().any():
        raise DataContractViolation(
            "checkpoint group contains more than one row for a target SKU"
        )
    source = frame.copy()
    source["sku_code"] = source["sku_code"].astype(str)
    if source["sku_code"].duplicated().any():
        raise DataContractViolation(
            "feature summary is not unique by target sku_code"
        )
    result = keys.merge(
        source,
        on="sku_code",
        how="left",
        validate="one_to_one",
        indicator="__feature_merge",
    )
    missing = result["__feature_merge"].ne("both")
    if missing.any():
        raise DataContractViolation(
            "feature output is missing scheduled target SKUs: "
            f"{result.loc[missing, 'sku_code'].tolist()}"
        )
    result = result.drop(columns="__feature_merge")
    scheduled = pd.to_datetime(result["decision_cutoff_time"], errors="coerce", utc=True)
    if "feature_cutoff_time" not in result:
        raise DataContractViolation("feature output has no feature_cutoff_time")
    actual = pd.to_datetime(result["feature_cutoff_time"], errors="coerce", utc=True)
    if scheduled.isna().any() or actual.isna().any() or actual.ne(scheduled).any():
        raise DataContractViolation(
            "feature output cutoff does not equal its scheduled checkpoint"
        )
    return result


def checkpoint_group_token(cutoff_time: datetime) -> str:
    rendered = cutoff_time.isoformat()
    readable = cutoff_time.strftime("%Y%m%dT%H%M%S")
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{digest}"


def replay_fingerprint(
    *,
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    options: Mapping[str, Any],
) -> str:
    payload = {
        "targets": targets.drop(columns=["__cutoff_utc"], errors="ignore")
        .sort_values("prediction_checkpoint_key")
        .to_dict(orient="records"),
        "candidates": candidates.sort_values("sku_code").to_dict(orient="records"),
        "options": dict(options),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def unique_cutoff_groups(
    schedule: pd.DataFrame,
) -> Iterable[tuple[datetime, pd.DataFrame]]:
    require_columns(schedule, ("__cutoff_utc",), "normalized replay schedule")
    for _, group in schedule.groupby("__cutoff_utc", sort=True):
        cutoff = _parse_aware(
            group.iloc[0]["decision_cutoff_time"], field="decision_cutoff_time"
        )
        yield cutoff, group.drop(columns="__cutoff_utc").reset_index(drop=True)
