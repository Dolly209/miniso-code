from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re
import unicodedata

import pandas as pd

from data.domestic import DataContractViolation
from data.domain import require_columns


SOCIAL_IP_ALIAS_COLUMNS = (
    "mapping_version",
    "catalog_as_of",
    "standard_ip_code",
    "standard_ip_name",
    "social_ip_name",
    "match_method",
    "shared_social_bucket",
    "enabled",
    "source_note",
)


def normalize_social_ip_name(value: object) -> str:
    """Normalize one standard/social IP label for deterministic lookup."""

    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def build_sku_social_aliases(
    standard_ip_mapping: pd.DataFrame,
    social_ip_alias_mapping: pd.DataFrame | None = None,
) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    """Resolve reviewed standard-IP memberships to normalized social labels.

    Both the comparable algorithm and point-in-time audits use this single
    data-layer helper so alias behavior cannot drift between stages.
    """

    require_columns(
        standard_ip_mapping,
        (
            "sku_code",
            "standard_ip_name",
            "standard_ip_level1",
            "standard_ip_level2",
        ),
        "standard IP mapping",
    )
    frame = standard_ip_mapping.copy()
    if "path1_eligible" in frame:
        eligible = pd.to_numeric(
            frame["path1_eligible"], errors="coerce"
        ).fillna(0).eq(1)
        frame = frame.loc[eligible]

    configured_by_code: dict[str, list[tuple[str, str]]] = {}
    if social_ip_alias_mapping is not None and not social_ip_alias_mapping.empty:
        require_columns(
            social_ip_alias_mapping,
            ("standard_ip_code", "social_ip_name", "match_method", "enabled"),
            "social IP alias mapping",
        )
        configured = social_ip_alias_mapping.loc[
            pd.to_numeric(
                social_ip_alias_mapping["enabled"], errors="coerce"
            ).fillna(0).eq(1)
        ]
        for row in configured.itertuples(index=False):
            code = str(row.standard_ip_code).strip()
            key = normalize_social_ip_name(row.social_ip_name)
            if code and key:
                configured_by_code.setdefault(code, []).append(
                    (key, str(row.match_method).strip())
                )

    aliases: dict[str, set[str]] = {}
    provenance: dict[str, dict[str, set[str]]] = {}
    for row in frame.itertuples(index=False):
        sku_code = str(row.sku_code).strip()
        if not sku_code:
            continue
        for column in (
            "standard_ip_name",
            "standard_ip_level1",
            "standard_ip_level2",
        ):
            key = normalize_social_ip_name(getattr(row, column))
            if not key:
                continue
            aliases.setdefault(sku_code, set()).add(key)
            provenance.setdefault(sku_code, {}).setdefault(key, set()).add(
                f"direct:{column}"
            )
        standard_ip_code = str(
            getattr(row, "standard_ip_code", "")
        ).strip()
        for key, match_method in configured_by_code.get(standard_ip_code, []):
            aliases.setdefault(sku_code, set()).add(key)
            provenance.setdefault(sku_code, {}).setdefault(key, set()).add(
                f"configured_csv:{match_method}"
            )
    return aliases, provenance


def _strip_text(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = result[column].fillna("").astype(str).str.strip()
    return result


def load_social_ip_alias_mapping(path: Path) -> pd.DataFrame:
    """Load the reviewed standard-IP to social-IP-name crosswalk.

    This file is deliberately separate from algorithm code.  It may map more
    than one standard IP to one social bucket only when every affected row is
    explicitly marked ``shared_social_bucket``.
    """

    if path.suffix.lower() != ".csv":
        raise DataContractViolation(
            f"social IP alias mapping must be a CSV artifact: {path}"
        )
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except FileNotFoundError as exc:
        raise DataContractViolation(
            f"social IP alias mapping not found: {path}"
        ) from exc
    require_columns(frame, SOCIAL_IP_ALIAS_COLUMNS, "social IP alias mapping")
    frame = _strip_text(frame, SOCIAL_IP_ALIAS_COLUMNS)

    enabled = pd.to_numeric(frame["enabled"], errors="coerce")
    shared = pd.to_numeric(frame["shared_social_bucket"], errors="coerce")
    if enabled.isna().any() or ~enabled.isin([0, 1]).all():
        raise DataContractViolation(
            "social IP alias enabled must contain only 0 or 1"
        )
    if shared.isna().any() or ~shared.isin([0, 1]).all():
        raise DataContractViolation(
            "social IP alias shared_social_bucket must contain only 0 or 1"
        )
    frame["enabled"] = enabled.astype(int)
    frame["shared_social_bucket"] = shared.astype(int)

    active = frame.loc[frame["enabled"].eq(1)].copy()
    required_nonempty = (
        "mapping_version",
        "catalog_as_of",
        "standard_ip_code",
        "standard_ip_name",
        "social_ip_name",
        "match_method",
    )
    for column in required_nonempty:
        if active[column].eq("").any():
            raise DataContractViolation(
                f"enabled social IP aliases require non-empty {column}"
            )
    versions = sorted(set(active["mapping_version"]))
    if len(versions) != 1:
        raise DataContractViolation(
            "enabled social IP aliases must use one mapping_version; "
            f"found={versions}"
        )
    invalid_dates = pd.to_datetime(
        active["catalog_as_of"], format="%Y-%m-%d", errors="coerce"
    ).isna()
    if invalid_dates.any():
        raise DataContractViolation(
            "social IP alias catalog_as_of must use YYYY-MM-DD"
        )
    duplicated = active.duplicated(
        ["standard_ip_code", "social_ip_name"], keep=False
    )
    if duplicated.any():
        examples = active.loc[
            duplicated, ["standard_ip_code", "social_ip_name"]
        ].head(5)
        raise DataContractViolation(
            "duplicate standard-IP/social-IP aliases: "
            + str(list(examples.itertuples(index=False, name=None)))
        )

    shared_counts = active.groupby("social_ip_name")["standard_ip_code"].nunique()
    repeated_social_names = set(shared_counts.loc[shared_counts.gt(1)].index)
    if repeated_social_names:
        repeated = active.loc[active["social_ip_name"].isin(repeated_social_names)]
        invalid_shared = repeated.loc[
            repeated["shared_social_bucket"].ne(1)
            | repeated["match_method"].ne("shared_social_bucket")
        ]
        if not invalid_shared.empty:
            examples = invalid_shared[
                ["standard_ip_code", "social_ip_name", "match_method"]
            ].head(5)
            raise DataContractViolation(
                "a social IP name mapped to multiple standard IPs must be "
                "explicitly marked shared_social_bucket: "
                + str(list(examples.itertuples(index=False, name=None)))
            )

    return frame.sort_values(
        ["enabled", "standard_ip_code", "social_ip_name"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
