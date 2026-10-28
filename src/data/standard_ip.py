from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import unicodedata

import pandas as pd

from data.domestic import DataContractViolation
from data.domain import normalize_sku_column, require_columns


STANDARD_IP_COLUMNS = (
    "sku_code",
    "sku_name",
    "standard_ip_level1",
    "standard_ip_level2",
    "standard_ip_code",
    "standard_ip_name",
    "mapping_method",
    "mapping_confidence",
    "path1_eligible",
    "text_ip_candidate",
    "evidence_field",
    "evidence_text",
)


@dataclass(frozen=True)
class ServerIpMappingResult:
    mapping: pd.DataFrame
    audit: pd.DataFrame


@dataclass(frozen=True)
class TaxonomyTextMatch:
    standard_ip_code: str
    score: int
    alias: str
    evidence_field: str
    evidence_text: str


def _strip_text(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = result[column].fillna("").astype(str).str.strip()
    return result


def _normalized_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return "".join(character for character in normalized if character.isalnum())


def _usable_lookup_value(value: object) -> bool:
    token = _normalized_text(value)
    return bool(token) and token not in {"无", "非版权", "none", "null", "nan"} and set(token) != {"0"}


def _unique_lookup(
    frame: pd.DataFrame,
    *,
    source_column: str,
    value_validator: Any | None = None,
) -> dict[str, str]:
    """Map a normalized source value to one unambiguous standard IP code."""

    if source_column not in frame:
        return {}
    validator = value_validator or _usable_lookup_value
    pairs: dict[str, set[str]] = {}
    for source_value, ip_code in frame[[source_column, "standard_ip_code"]].itertuples(
        index=False, name=None
    ):
        if not validator(source_value):
            continue
        key = _normalized_text(source_value)
        pairs.setdefault(key, set()).add(str(ip_code))
    return {key: next(iter(values)) for key, values in pairs.items() if len(values) == 1}


def load_standard_ip_alias_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataContractViolation(f"standard IP alias config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataContractViolation(f"invalid standard IP alias config: {path}: {exc}") from exc
    required = {
        "blocked_text_codes",
        "redirects",
        "parent_defaults",
        "parent_aliases",
        "canonical_aliases",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise DataContractViolation(
            "standard IP alias config is missing keys: " + ", ".join(missing)
        )
    return payload


def _has_explicit_collaboration_marker(text: str) -> bool:
    return bool(
        "联名" in text
        or "×" in text
        or re.search(r"(?<=[\u4e00-\u9fff])[xX](?=[\u4e00-\u9fff])", text)
    )


def _taxonomy_terms(
    mapping: pd.DataFrame,
    alias_config: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    dict[str, set[str]],
    set[str],
    dict[str, set[str]],
    dict[str, int],
]:
    """Rebuild the canonical vocabulary represented by the workbook 统计 sheet.

    ``excel_exact`` rows contain exactly the same 301 unique level-1/level-2
    combinations as the pivot-style 统计 sheet. Keeping the normalized CSV as
    the server input avoids an Excel dependency while preserving that taxonomy.
    """

    taxonomy = mapping.loc[
        mapping["mapping_method"].eq("excel_exact")
        & mapping["path1_eligible"].eq(1)
        & mapping["standard_ip_code"].ne(""),
        [
            "standard_ip_code",
            "standard_ip_level1",
            "standard_ip_level2",
            "standard_ip_name",
        ],
    ].drop_duplicates("standard_ip_code")
    if taxonomy.empty:
        raise DataContractViolation("standard IP reference has no Excel taxonomy rows")
    metadata = taxonomy.set_index("standard_ip_code")
    known_codes = set(metadata.index.astype(str))

    redirects = {
        str(source): str(target)
        for source, target in dict(alias_config.get("redirects", {})).items()
    }
    parent_defaults = {
        str(parent): str(code)
        for parent, code in dict(alias_config.get("parent_defaults", {})).items()
    }
    configured_aliases = {
        str(code): [str(alias) for alias in aliases]
        for code, aliases in dict(alias_config.get("canonical_aliases", {})).items()
    }
    unknown_targets = sorted(
        (set(redirects.values()) | set(parent_defaults.values()) | set(configured_aliases))
        - known_codes
    )
    if unknown_targets:
        raise DataContractViolation(
            "alias config targets are absent from 统计 taxonomy: "
            + ", ".join(unknown_targets)
        )

    blocked = {str(value) for value in alias_config.get("blocked_text_codes", [])}
    aliases_by_code: dict[str, set[str]] = {code: set() for code in known_codes}
    for row in taxonomy.itertuples(index=False):
        code = str(row.standard_ip_code)
        if code in blocked or code in redirects:
            continue
        canonical = str(row.standard_ip_level2 or row.standard_ip_level1).strip()
        if canonical:
            aliases_by_code[code].add(canonical)

    # Redirect duplicate/synonymous labels in the statistics sheet to one
    # canonical style without changing Excel-exact SKU assignments.
    for source, target in redirects.items():
        if source not in known_codes:
            raise DataContractViolation(
                f"alias redirect source is absent from 统计 taxonomy: {source}"
            )
        source_row = metadata.loc[source]
        for value in (
            source_row["standard_ip_level1"],
            source_row["standard_ip_level2"],
            source_row["standard_ip_name"],
        ):
            if str(value).strip():
                aliases_by_code[target].add(str(value).strip())

    for parent, target in parent_defaults.items():
        aliases_by_code[target].add(parent)
        for alias in dict(alias_config.get("parent_aliases", {})).get(parent, []):
            aliases_by_code[target].add(str(alias))
    for code, aliases in configured_aliases.items():
        aliases_by_code[code].update(aliases)

    normalized_aliases: dict[str, set[str]] = {}
    for code, aliases in aliases_by_code.items():
        for alias in aliases:
            token = _normalized_text(alias)
            if len(token) >= 2:
                normalized_aliases.setdefault(token, set()).add(code)

    parent_tokens: dict[str, set[str]] = {}
    for parent in set(metadata["standard_ip_level1"].astype(str)):
        token = _normalized_text(parent)
        if token:
            parent_tokens.setdefault(parent, set()).add(token)
    for parent, aliases in dict(alias_config.get("parent_aliases", {})).items():
        for alias in aliases:
            token = _normalized_text(alias)
            if token:
                parent_tokens.setdefault(str(parent), set()).add(token)

    precedence = {
        str(parent): rank
        for rank, parent in enumerate(
            alias_config.get("duplicate_child_parent_precedence", [])
        )
    }
    return (
        metadata,
        normalized_aliases,
        set(parent_defaults.values()),
        parent_tokens,
        precedence,
    )


def _match_taxonomy_text(
    product: pd.Series,
    *,
    metadata: pd.DataFrame,
    aliases: Mapping[str, set[str]],
    generic_codes: set[str],
    parent_tokens: Mapping[str, set[str]],
    duplicate_child_parent_precedence: Mapping[str, int],
) -> tuple[list[TaxonomyTextMatch], list[str], bool]:
    field_weights = {"spu_name": 300, "sku_name": 200, "copyright_name": 100}
    candidates: dict[str, TaxonomyTextMatch] = {}
    raw_texts: list[str] = []
    searchable_fields: list[tuple[str, int, str, str]] = []
    for field, field_weight in field_weights.items():
        value = product.get(field, "")
        raw = "" if value is None or pd.isna(value) else str(value).strip()
        if not _usable_lookup_value(raw):
            continue
        raw_texts.append(raw)
        searchable_fields.append((field, field_weight, raw, _normalized_text(raw)))
    combined_normalized = "|".join(item[3] for item in searchable_fields)

    for field, field_weight, raw, normalized in searchable_fields:
        for alias, codes in aliases.items():
            if alias not in normalized:
                continue
            for code in codes:
                meta = metadata.loc[code]
                has_level2 = bool(str(meta["standard_ip_level2"]).strip())
                score = field_weight + min(len(alias), 24) * 4
                if has_level2:
                    score += 60
                if code in generic_codes:
                    score -= 80
                parent = str(meta["standard_ip_level1"])
                has_parent_cue = any(
                    token in combined_normalized
                    for token in parent_tokens.get(parent, set())
                )
                if has_level2 and has_parent_cue:
                    score += 30
                match = TaxonomyTextMatch(code, score, alias, field, raw)
                previous = candidates.get(code)
                if previous is None or match.score > previous.score:
                    candidates[code] = match

    # A specific child suppresses a generic family/default under the same
    # level-1 IP (for example, 大耳狗 suppresses 三丽鸥家族).
    for generic_code in list(generic_codes):
        generic = candidates.get(generic_code)
        if generic is None:
            continue
        parent = str(metadata.loc[generic_code, "standard_ip_level1"])
        if any(
            code != generic_code
            and str(metadata.loc[code, "standard_ip_level1"]) == parent
            for code in candidates
        ):
            candidates.pop(generic_code, None)

    # The statistics sheet contains one duplicate child label: 玩具总动员 is
    # listed under both 迪士尼 and 皮克斯. Use explicit parent words found in
    # any product field to choose the hierarchy. If both parents are written
    # (the common "迪士尼皮克斯" style), the reviewed precedence selects
    # 皮克斯; without a parent cue the result remains ambiguous for review.
    duplicate_children: dict[str, list[str]] = {}
    for code in candidates:
        child = str(metadata.loc[code, "standard_ip_level2"]).strip()
        if child:
            duplicate_children.setdefault(child, []).append(code)
    for codes in duplicate_children.values():
        parents = {str(metadata.loc[code, "standard_ip_level1"]) for code in codes}
        if len(codes) < 2 or len(parents) < 2:
            continue
        cued = [
            code
            for code in codes
            if any(
                token in combined_normalized
                for token in parent_tokens.get(
                    str(metadata.loc[code, "standard_ip_level1"]), set()
                )
            )
        ]
        selected: str | None = None
        if len(cued) == 1:
            selected = cued[0]
        elif len(cued) > 1:
            ranked = sorted(
                cued,
                key=lambda code: duplicate_child_parent_precedence.get(
                    str(metadata.loc[code, "standard_ip_level1"]), 10_000
                ),
            )
            first_rank = duplicate_child_parent_precedence.get(
                str(metadata.loc[ranked[0], "standard_ip_level1"]), 10_000
            )
            if sum(
                duplicate_child_parent_precedence.get(
                    str(metadata.loc[code, "standard_ip_level1"]), 10_000
                )
                == first_rank
                for code in ranked
            ) == 1:
                selected = ranked[0]
        if selected:
            for code in codes:
                if code != selected:
                    candidates.pop(code, None)

    if not candidates:
        return [], [], False
    search_text = "|".join(raw_texts)
    collaboration = _has_explicit_collaboration_marker(search_text)
    best_score = max(match.score for match in candidates.values())
    if collaboration:
        # Keep the strongest specific match per top-level IP. A 40-point band
        # permits genuine cross-IP product names while excluding incidental
        # low-priority words in another field.
        by_parent: dict[str, TaxonomyTextMatch] = {}
        for code, match in candidates.items():
            if match.score < best_score - 40:
                continue
            parent = str(metadata.loc[code, "standard_ip_level1"])
            previous = by_parent.get(parent)
            if previous is None or match.score > previous.score:
                by_parent[parent] = match
        selected = sorted(by_parent.values(), key=lambda match: match.standard_ip_code)
        return selected, [], True

    best = [match for match in candidates.values() if match.score == best_score]
    if len(best) == 1:
        return best, [], False

    if all(match.standard_ip_code in generic_codes for match in best):
        ranked = sorted(
            best,
            key=lambda match: duplicate_child_parent_precedence.get(
                str(
                    metadata.loc[
                        match.standard_ip_code, "standard_ip_level1"
                    ]
                ),
                10_000,
            ),
        )
        first_rank = duplicate_child_parent_precedence.get(
            str(metadata.loc[ranked[0].standard_ip_code, "standard_ip_level1"]),
            10_000,
        )
        if sum(
            duplicate_child_parent_precedence.get(
                str(metadata.loc[match.standard_ip_code, "standard_ip_level1"]),
                10_000,
            )
            == first_rank
            for match in ranked
        ) == 1:
            return [ranked[0]], [], False

    # Within one parent family, the right-most equal-strength character name
    # is usually the concrete variant after the series name, for example
    # "米奇好友...（维尼熊）" or "米奇家族...（黛西）".
    best_parents = {
        str(metadata.loc[match.standard_ip_code, "standard_ip_level1"])
        for match in best
    }
    if len(best_parents) == 1 and len({match.evidence_field for match in best}) == 1:
        positioned = [
            (_normalized_text(match.evidence_text).rfind(match.alias), match)
            for match in best
        ]
        rightmost = max(position for position, _ in positioned)
        winners = [match for position, match in positioned if position == rightmost]
        if len(winners) == 1:
            return winners, [], False
    return [], sorted(match.standard_ip_code for match in best), False


def load_standard_ip_mapping(path: Path) -> pd.DataFrame:
    """Load the reviewed SKU-to-standard-IP reference artifact.

    The production server intentionally consumes CSV rather than the source
    workbook, so it does not need an Excel engine and algorithm modules never
    parse business-maintained spreadsheets directly.
    """

    if path.suffix.lower() != ".csv":
        raise DataContractViolation(
            f"standard IP mapping must be a normalized CSV artifact: {path}"
        )
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except FileNotFoundError as exc:
        raise DataContractViolation(f"standard IP mapping not found: {path}") from exc
    require_columns(frame, STANDARD_IP_COLUMNS, "standard IP mapping")
    frame = normalize_sku_column(frame)
    frame = _strip_text(
        frame,
        (
            "sku_name",
            "spu_code",
            "spu_name",
            "standard_ip_level1",
            "standard_ip_level2",
            "standard_ip_code",
            "standard_ip_name",
            "mapping_method",
            "text_ip_candidate",
            "evidence_field",
            "evidence_text",
            "copyright_name",
            "raw_theme_code",
            "raw_theme_category",
            "source_workbook",
            "source_sheet",
            "mapping_note",
        ),
    )
    confidence = pd.to_numeric(frame["mapping_confidence"], errors="coerce")
    eligible = pd.to_numeric(frame["path1_eligible"], errors="coerce")
    if confidence.isna().any() or ((confidence < 0) | (confidence > 1)).any():
        raise DataContractViolation("mapping_confidence must be numeric in [0, 1]")
    if eligible.isna().any() or ~eligible.isin([0, 1]).all():
        raise DataContractViolation("path1_eligible must contain only 0 or 1")
    frame["mapping_confidence"] = confidence.astype(float)
    frame["path1_eligible"] = eligible.astype(int)
    invalid_eligible = frame["path1_eligible"].eq(1) & frame["standard_ip_code"].eq("")
    if invalid_eligible.any():
        examples = frame.loc[invalid_eligible, "sku_code"].head(5).tolist()
        raise DataContractViolation(
            f"PATH1-eligible mappings lack standard_ip_code: {examples}"
        )
    duplicates = frame.loc[frame["standard_ip_code"].ne("")].duplicated(
        ["sku_code", "standard_ip_code"]
    )
    if duplicates.any():
        examples = frame.loc[
            frame["standard_ip_code"].ne("")
            & frame.duplicated(["sku_code", "standard_ip_code"], keep=False),
            ["sku_code", "standard_ip_code"],
        ].head(5)
        raise DataContractViolation(
            "duplicate SKU-standard-IP memberships: "
            + str(list(examples.itertuples(index=False, name=None)))
        )
    return frame.sort_values(
        ["sku_code", "standard_ip_code", "mapping_method"], kind="stable"
    ).reset_index(drop=True)


def validate_resolved_standard_ip_snapshot(
    mapping: pd.DataFrame,
    *,
    cohort: pd.DataFrame,
    report: Mapping[str, Any],
    decision_cutoff_time: datetime,
    reference_snapshot_time: datetime | None = None,
    allow_cohort_subset: bool = False,
) -> str:
    """Validate a materialized cohort SKU-to-IP snapshot before downstream use.

    M1 and later stages must look up this frozen artifact rather than rerunning
    text inference. Rebuilding is an explicit upstream action performed by
    ``build_standard_ip_mapping.py`` when the cohort, cutoff, reference, alias
    rules, or mapping implementation changes.

    Historical pre-launch replay deliberately separates the fact cutoff from
    the fixed product-master/reference snapshot confirmed by the business.  In
    that mode ``reference_snapshot_time`` is the fixed snapshot time and the
    replay cohort may be a strict subset of the mapping universe.  Missing
    cohort SKUs still fail; only unrelated extra mapping rows are allowed.
    """

    require_columns(
        mapping,
        (*STANDARD_IP_COLUMNS, "dataset_version", "mapping_status"),
        "resolved standard IP snapshot",
    )
    require_columns(cohort, ("sku_code", "dataset_version"), "Round 3 cohort")
    resolved = normalize_sku_column(mapping)
    cohort_frame = normalize_sku_column(cohort)
    if cohort_frame["sku_code"].duplicated().any():
        raise DataContractViolation("Round 3 cohort must contain one row per SKU")

    cohort_versions = {
        str(value).strip()
        for value in cohort_frame["dataset_version"]
        if pd.notna(value) and str(value).strip()
    }
    if len(cohort_versions) != 1:
        raise DataContractViolation(
            "Round 3 cohort must contain exactly one non-empty dataset_version"
        )
    dataset_version = next(iter(cohort_versions))

    mapping_versions = {
        str(value).strip()
        for value in resolved["dataset_version"]
        if pd.notna(value) and str(value).strip()
    }
    if mapping_versions != {dataset_version}:
        raise DataContractViolation(
            "resolved standard IP snapshot dataset_version does not match cohort: "
            f"cohort={dataset_version!r} snapshot={sorted(mapping_versions)!r}"
        )
    if resolved["dataset_version"].fillna("").astype(str).str.strip().eq("").any():
        raise DataContractViolation(
            "resolved standard IP snapshot contains blank dataset_version rows"
        )

    cohort_skus = set(cohort_frame["sku_code"])
    snapshot_skus = set(resolved["sku_code"])
    missing = sorted(cohort_skus - snapshot_skus)
    extra = sorted(snapshot_skus - cohort_skus)
    if missing or (extra and not allow_cohort_subset):
        raise DataContractViolation(
            "resolved standard IP snapshot SKU universe does not match cohort; "
            f"missing_count={len(missing)} missing_examples={missing[:5]} "
            f"extra_count={len(extra)} extra_examples={extra[:5]}"
        )

    if str(report.get("spec_version")) != "3.2":
        raise DataContractViolation(
            "standard IP snapshot report must declare spec_version=3.2"
        )
    if report.get("stage") != "standard IP mapping":
        raise DataContractViolation(
            "standard IP snapshot report has an unexpected stage"
        )
    if report.get("status") != "implemented_output":
        raise DataContractViolation(
            "standard IP snapshot report is not an implemented output"
        )
    try:
        report_cutoff = datetime.fromisoformat(
            str(report["decision_cutoff_time"]).replace("Z", "+00:00")
        )
    except (KeyError, ValueError) as exc:
        raise DataContractViolation(
            "standard IP snapshot report has an invalid decision_cutoff_time"
        ) from exc
    if report_cutoff.tzinfo is None or report_cutoff.utcoffset() is None:
        raise DataContractViolation(
            "standard IP snapshot report cutoff must include a UTC offset"
        )
    expected_snapshot_time = reference_snapshot_time or decision_cutoff_time
    if (
        expected_snapshot_time.tzinfo is None
        or expected_snapshot_time.utcoffset() is None
    ):
        raise DataContractViolation(
            "standard IP reference snapshot time must include a UTC offset"
        )
    if report_cutoff.astimezone(timezone.utc) != expected_snapshot_time.astimezone(
        timezone.utc
    ):
        raise DataContractViolation(
            "standard IP snapshot cutoff does not match the requested reference "
            "snapshot: "
            f"snapshot={report_cutoff.isoformat()} "
            f"reference={expected_snapshot_time.isoformat()}"
        )
    expected_partition = expected_snapshot_time.date().isoformat()
    if str(report.get("source_product_partition")) != expected_partition:
        raise DataContractViolation(
            "standard IP snapshot product partition does not match cutoff date: "
            f"snapshot={report.get('source_product_partition')!r} "
            f"expected={expected_partition!r}"
        )

    expected_report_skus = len(snapshot_skus) if allow_cohort_subset else len(cohort_skus)
    if int(report.get("cohort_skus", -1)) != expected_report_skus:
        raise DataContractViolation(
            "standard IP snapshot report cohort_skus does not match its mapping "
            "universe"
        )
    eligible = resolved.loc[
        resolved["path1_eligible"].eq(1)
        & resolved["standard_ip_code"].ne("")
    ]
    mapped_skus = int(eligible["sku_code"].nunique())
    if int(report.get("mapped_cohort_skus", -1)) != mapped_skus:
        raise DataContractViolation(
            "standard IP snapshot report mapped_cohort_skus does not match CSV"
        )
    return dataset_version


def path1_ip_membership(
    mapping: pd.DataFrame, *, product_skus: Iterable[object] | None = None
) -> pd.DataFrame:
    """Return the canonical set-membership frame consumed by M1.

    Text candidates that still need review and unmapped rows are deliberately
    excluded. Missing mappings therefore cannot silently fall back to the
    coarse display-object ``ip_code`` delivered by the source product table.
    """

    require_columns(mapping, STANDARD_IP_COLUMNS, "standard IP mapping")
    result = mapping.loc[
        mapping["path1_eligible"].eq(1) & mapping["standard_ip_code"].ne(""),
        [
            "sku_code",
            "standard_ip_code",
            "standard_ip_name",
            "mapping_method",
            "mapping_confidence",
        ],
    ].copy()
    if product_skus is not None:
        allowed = {str(value).strip() for value in product_skus if str(value).strip()}
        result = result.loc[result["sku_code"].isin(allowed)]
    result = result.rename(
        columns={"standard_ip_code": "ip_code", "standard_ip_name": "ip_name"}
    )
    return result.drop_duplicates(["sku_code", "ip_code"]).reset_index(drop=True)


def build_server_ip_mapping(
    *,
    products: pd.DataFrame,
    raw_ip_membership: pd.DataFrame,
    mapping: pd.DataFrame,
    alias_config: Mapping[str, Any] | None = None,
) -> ServerIpMappingResult:
    """Align the reviewed reference with one server product snapshot by SKU.

    Exact SKU matches always win. For server SKUs absent from the reviewed
    reference, conservative inference searches product text against the
    workbook 统计 taxonomy, then uses unambiguous copyright/SPU evidence to
    resolve or supplement it. Every inferred row retains its method and
    evidence; unresolved conflicts are excluded from PATH1.
    """

    require_columns(products, ("sku_code", "sku_name"), "server products")
    require_columns(raw_ip_membership, ("sku_code", "ip_code"), "raw IP membership")
    products = normalize_sku_column(products)
    if products["sku_code"].duplicated().any():
        raise DataContractViolation("server products must be one row per SKU")
    raw = raw_ip_membership.copy()
    raw["sku_code"] = raw["sku_code"].astype(str).str.strip()
    raw["ip_code"] = raw["ip_code"].fillna("").astype(str).str.strip()
    raw_codes = (
        raw.loc[raw["ip_code"].ne("")]
        .groupby("sku_code")["ip_code"]
        .agg(lambda values: "|".join(sorted(set(values))))
        .rename("raw_proxy_ip_codes")
    )

    server_skus = set(products["sku_code"])
    aligned = mapping.loc[mapping["sku_code"].isin(server_skus)].copy()
    aligned = aligned.merge(
        products[["sku_code", "sku_name"]].rename(columns={"sku_name": "server_sku_name"}),
        on="sku_code",
        how="left",
        validate="many_to_one",
    )

    alias_config = dict(alias_config or {})
    reviewed = mapping.loc[
        mapping["path1_eligible"].eq(1) & mapping["standard_ip_code"].ne("")
    ].copy()
    (
        metadata,
        taxonomy_aliases,
        generic_codes,
        parent_tokens,
        duplicate_child_parent_precedence,
    ) = _taxonomy_terms(mapping, alias_config)
    minimum_numeric_spu_length = int(
        alias_config.get("minimum_numeric_spu_length", 8)
    )

    def valid_spu_code(value: object) -> bool:
        if not _usable_lookup_value(value):
            return False
        token = _normalized_text(value)
        return not token.isdigit() or len(token) >= minimum_numeric_spu_length

    spu_code_lookup = _unique_lookup(
        reviewed,
        source_column="spu_code",
        value_validator=valid_spu_code,
    )
    spu_name_lookup = _unique_lookup(reviewed, source_column="spu_name")
    copyright_lookup = _unique_lookup(reviewed, source_column="copyright_name")
    redirects = {
        str(source): str(target)
        for source, target in dict(alias_config.get("redirects", {})).items()
    }
    # Exact server fields can point at an older/synonymous code from the
    # workbook (for example TOM and JERRY). Apply the same canonical redirect
    # used by text inference so one SKU cannot regain a deprecated synonym.
    for lookup in (spu_code_lookup, spu_name_lookup, copyright_lookup):
        for key, code in list(lookup.items()):
            lookup[key] = redirects.get(code, code)

    directly_mapped_skus = set(
        aligned.loc[
            aligned["path1_eligible"].eq(1) & aligned["standard_ip_code"].ne(""),
            "sku_code",
        ]
    )
    reference_source = (
        str(mapping["source_workbook"].iloc[0])
        if "source_workbook" in mapping and not mapping.empty
        else ""
    )
    raw_code_by_sku = raw_codes.to_dict()
    inferred_rows: list[dict[str, object]] = []

    def display_text(value: object) -> str:
        return "" if value is None or pd.isna(value) else str(value)

    def add_inference(
        *,
        product: pd.Series,
        standard_ip_code: str,
        method: str,
        confidence: float,
        evidence_field: str,
        evidence_text: str,
        note: str,
        found: dict[str, dict[str, object]],
    ) -> None:
        if standard_ip_code not in metadata.index:
            return
        previous = found.get(standard_ip_code)
        if previous is not None and float(previous["mapping_confidence"]) >= confidence:
            return
        ip = metadata.loc[standard_ip_code]
        found[standard_ip_code] = {
            "sku_code": str(product["sku_code"]),
            "sku_name": display_text(product.get("sku_name", "")),
            "spu_code": display_text(product.get("spu_code", "")),
            "spu_name": display_text(product.get("spu_name", "")),
            "standard_ip_level1": str(ip["standard_ip_level1"]),
            "standard_ip_level2": str(ip["standard_ip_level2"]),
            "standard_ip_code": standard_ip_code,
            "standard_ip_name": str(ip["standard_ip_name"]),
            "mapping_method": method,
            "mapping_confidence": confidence,
            "path1_eligible": 1,
            "text_ip_candidate": str(ip["standard_ip_name"]),
            "evidence_field": evidence_field,
            "evidence_text": evidence_text,
            "copyright_name": display_text(product.get("copyright_name", "")),
            "raw_theme_code": raw_code_by_sku.get(str(product["sku_code"]), ""),
            "raw_theme_category": "",
            "source_workbook": reference_source,
            "source_sheet": "server_product_snapshot",
            "source_row_count": 1,
            "mapping_note": note,
            "server_sku_name": display_text(product.get("sku_name", "")),
        }

    def add_conflict(
        *,
        product: pd.Series,
        candidates: Iterable[str],
        evidence_text: str,
        note: str,
        found: dict[str, dict[str, object]],
    ) -> None:
        candidate_codes = sorted(
            code for code in set(candidates) if code in metadata.index
        )
        candidate_names = [
            str(metadata.loc[code, "standard_ip_name"]) for code in candidate_codes
        ]
        found["__conflict__"] = {
            "sku_code": str(product["sku_code"]),
            "sku_name": display_text(product.get("sku_name", "")),
            "spu_code": display_text(product.get("spu_code", "")),
            "spu_name": display_text(product.get("spu_name", "")),
            "standard_ip_level1": "",
            "standard_ip_level2": "",
            "standard_ip_code": "",
            "standard_ip_name": "",
            "mapping_method": "server_inference_conflict",
            "mapping_confidence": 0.0,
            "path1_eligible": 0,
            "text_ip_candidate": "|".join(candidate_names),
            "evidence_field": "sku_name|spu_name|copyright_name",
            "evidence_text": evidence_text,
            "copyright_name": display_text(product.get("copyright_name", "")),
            "raw_theme_code": raw_code_by_sku.get(str(product["sku_code"]), ""),
            "raw_theme_category": "",
            "source_workbook": reference_source,
            "source_sheet": "server_product_snapshot",
            "source_row_count": 1,
            "mapping_note": note + " 候选=" + "|".join(candidate_codes),
            "server_sku_name": display_text(product.get("sku_name", "")),
        }

    for _, product in products.loc[
        ~products["sku_code"].isin(directly_mapped_skus)
    ].iterrows():
        found: dict[str, dict[str, object]] = {}
        text_matches, ambiguous_text_codes, collaboration = _match_taxonomy_text(
            product,
            metadata=metadata,
            aliases=taxonomy_aliases,
            generic_codes=generic_codes,
            parent_tokens=parent_tokens,
            duplicate_child_parent_precedence=duplicate_child_parent_precedence,
        )

        exact_codes: dict[str, str | None] = {}
        for field, lookup in (
            ("copyright_name", copyright_lookup),
            ("spu_code", spu_code_lookup),
            ("spu_name", spu_name_lookup),
        ):
            value = product.get(field, "")
            valid = valid_spu_code(value) if field == "spu_code" else _usable_lookup_value(value)
            exact_codes[field] = (
                lookup.get(_normalized_text(value)) if valid else None
            )

        if ambiguous_text_codes:
            resolver = next(
                (
                    exact_codes[field]
                    for field in ("copyright_name", "spu_code", "spu_name")
                    if exact_codes[field] in ambiguous_text_codes
                ),
                None,
            )
            if resolver:
                add_inference(
                    product=product,
                    standard_ip_code=resolver,
                    method="server_inferred_taxonomy_resolved",
                    confidence=0.93,
                    evidence_field="统计词表+精确字段",
                    evidence_text="|".join(
                        display_text(product.get(field, ""))
                        for field in ("sku_name", "spu_name", "copyright_name")
                    ),
                    note="统计词表出现多个候选，由服务器精确字段消歧。",
                    found=found,
                )
            else:
                add_conflict(
                    product=product,
                    candidates=ambiguous_text_codes,
                    evidence_text="|".join(
                        display_text(product.get(field, ""))
                        for field in ("sku_name", "spu_name", "copyright_name")
                    ),
                    note="统计词表文字命中多个同分候选，且没有可靠字段消歧。",
                    found=found,
                )
        elif text_matches:
            text_codes = {match.standard_ip_code for match in text_matches}
            copyright_code = exact_codes["copyright_name"]
            if copyright_code and copyright_code not in text_codes and not collaboration:
                add_inference(
                    product=product,
                    standard_ip_code=copyright_code,
                    method="server_inferred_taxonomy_resolved",
                    confidence=0.92,
                    evidence_field="版权名称+统计词表",
                    evidence_text="|".join(
                        display_text(product.get(field, ""))
                        for field in ("sku_name", "spu_name", "copyright_name")
                    ),
                    note=(
                        "文字字段存在冲突，唯一版权名称映射优先于SPU/普通标题词；"
                        "被排除候选=" + "|".join(sorted(text_codes)) + "。"
                    ),
                    found=found,
                )
            else:
                for match in text_matches:
                    add_inference(
                        product=product,
                        standard_ip_code=match.standard_ip_code,
                        method="server_inferred_taxonomy_text",
                        confidence=0.93 if len(text_matches) == 1 else 0.90,
                        evidence_field=match.evidence_field,
                        evidence_text=match.evidence_text,
                        note=(
                            "服务器文字命中Excel“统计”Sheet标准IP层级；"
                            f"匹配词={match.alias}。"
                        ),
                        found=found,
                    )
        else:
            for field, method, confidence in (
                ("copyright_name", "server_inferred_copyright_name", 0.94),
                ("spu_code", "server_inferred_spu_code", 0.92),
                ("spu_name", "server_inferred_spu_name", 0.90),
            ):
                code = exact_codes[field]
                if not code:
                    continue
                add_inference(
                    product=product,
                    standard_ip_code=code,
                    method=method,
                    confidence=confidence,
                    evidence_field=field,
                    evidence_text=display_text(product.get(field, "")),
                    note=f"服务器{field}与已审核映射中的唯一标准IP精确匹配。",
                    found=found,
                )
                break
        inferred_rows.extend(found.values())

    if inferred_rows:
        aligned = pd.concat([aligned, pd.DataFrame(inferred_rows)], ignore_index=True, sort=False)
        priority = {
            "excel_exact": 0,
            "text_inferred_reference_alias": 1,
            "server_inferred_taxonomy_text": 2,
            "server_inferred_taxonomy_resolved": 3,
            "server_inferred_copyright_name": 4,
            "server_inferred_spu_code": 5,
            "server_inferred_spu_name": 6,
            "server_inference_conflict": 7,
            "text_candidate_needs_review": 8,
            "unmapped": 9,
        }
        aligned["__method_priority"] = aligned["mapping_method"].map(priority).fillna(99)
        aligned = (
            aligned.sort_values(
                ["sku_code", "standard_ip_code", "__method_priority"], kind="stable"
            )
            .drop_duplicates(["sku_code", "standard_ip_code"], keep="first")
            .drop(columns="__method_priority")
        )

    leading = [
        "sku_code",
        "server_sku_name",
        "sku_name",
        "standard_ip_level1",
        "standard_ip_level2",
        "standard_ip_code",
        "standard_ip_name",
        "mapping_method",
        "mapping_confidence",
        "path1_eligible",
        "text_ip_candidate",
        "evidence_field",
        "evidence_text",
    ]
    aligned = aligned[leading + [column for column in aligned.columns if column not in leading]]

    def summarize(group: pd.DataFrame) -> pd.Series:
        eligible = group.loc[
            group["path1_eligible"].eq(1) & group["standard_ip_code"].ne("")
        ]
        methods = set(group["mapping_method"])
        if "excel_exact" in methods and not eligible.empty:
            status = "mapped_excel"
        elif not eligible.empty:
            status = "mapped_text_inferred"
        elif "server_inference_conflict" in methods:
            status = "inference_conflict"
        elif group["text_ip_candidate"].ne("").any():
            status = "candidate_needs_review"
        else:
            status = "unmapped_reference"
        return pd.Series(
            {
                "mapping_status": status,
                "standard_ip_count": int(eligible["standard_ip_code"].nunique()),
                "standard_ip_codes": "|".join(
                    sorted(set(eligible["standard_ip_code"]))
                ),
                "mapping_methods": "|".join(sorted(methods)),
                "text_ip_candidates": "|".join(
                    sorted(set(group.loc[group["text_ip_candidate"].ne(""), "text_ip_candidate"]))
                ),
            }
        )

    if aligned.empty:
        summary = pd.DataFrame(
            columns=[
                "sku_code",
                "mapping_status",
                "standard_ip_count",
                "standard_ip_codes",
                "mapping_methods",
                "text_ip_candidates",
            ]
        )
    else:
        summary_rows = []
        for sku_code, group in aligned.groupby("sku_code", sort=False):
            row = summarize(group).to_dict()
            row["sku_code"] = sku_code
            summary_rows.append(row)
        summary = pd.DataFrame(summary_rows)

    audit_columns = [
        column
        for column in (
            "sku_code",
            "sku_name",
            "spu_code",
            "spu_name",
            "copyright_code",
            "copyright_name",
            "copyright_category",
            "manage_cate_name",
        )
        if column in products.columns
    ]
    audit = products[audit_columns].copy()
    audit = audit.merge(summary, on="sku_code", how="left", validate="one_to_one")
    audit["mapping_status"] = audit["mapping_status"].fillna("not_in_reference")
    audit["standard_ip_count"] = (
        pd.to_numeric(audit["standard_ip_count"], errors="coerce").fillna(0).astype(int)
    )
    for column in (
        "standard_ip_codes",
        "mapping_methods",
        "text_ip_candidates",
    ):
        audit[column] = audit[column].fillna("")
    audit = audit.merge(raw_codes, on="sku_code", how="left", validate="one_to_one")
    audit["raw_proxy_ip_codes"] = audit["raw_proxy_ip_codes"].fillna("")
    return ServerIpMappingResult(
        mapping=aligned.sort_values(
            ["sku_code", "standard_ip_code", "mapping_method"], kind="stable"
        ).reset_index(drop=True),
        audit=audit.sort_values("sku_code", kind="stable").reset_index(drop=True),
    )
