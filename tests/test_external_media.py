from __future__ import annotations

from datetime import date

import pandas as pd

from comparable.external_media import apply_external_media_correction
from data.domestic import DataContractViolation
from data.social_ip_aliases import load_social_ip_alias_mapping


def _social_row(day: str, ip: str, post: float, interaction: float) -> dict[str, object]:
    return {
        "stat_date": day,
        "ip_name": ip,
        "platform_code": "douyin",
        "post_count": post,
        "interaction_count": interaction,
        "unique_author_count": post,
        "koc_author_count": post / 2,
        "kol_author_count": 1,
        "view_count": interaction * 10,
        "positive_post_share": 0.8,
        "negative_post_share": 0.1,
        "text_coverage_rate": 0.9,
        "view_coverage_rate": 0.9,
    }


def test_external_media_is_bounded_relative_direction_only() -> None:
    summary = pd.DataFrame([{"sku_code": "T"}])
    candidates = pd.DataFrame(
        [{
            "target_sku_code": "T",
            "candidate_sku_code": "C",
            "selected": 1,
            "alpha_weight": 1.0,
        }]
    )
    targets = pd.DataFrame(
        [{"sku_code": "T", "first_actual_sale_date": "2026-02-01"}]
    )
    mapping = pd.DataFrame(
        [
            {
                "sku_code": "T",
                "standard_ip_name": "热门IP",
                "standard_ip_level1": "热门IP",
                "standard_ip_level2": "",
                "path1_eligible": 1,
            },
            {
                "sku_code": "C",
                "standard_ip_name": "普通IP",
                "standard_ip_level1": "普通IP",
                "standard_ip_level2": "",
                "path1_eligible": 1,
            },
        ]
    )
    media = pd.DataFrame(
        [
            _social_row("2026-01-02", "热门IP", 100, 1000),
            _social_row("2026-01-31", "热门IP", 300, 5000),
            _social_row("2026-01-02", "普通IP", 10, 100),
            _social_row("2026-01-31", "普通IP", 12, 120),
        ]
    )
    config = {
        "external_coefficient_bounds": [0.7, 1.3],
        "external_media": {
            "prelaunch_days": 30,
            "sentiment_coverage_min": 0.5,
            "minimum_snapshot_days_for_trend": 2,
            "coefficient_slope": 0.15,
            "score_difference_bounds": [-2.0, 2.0],
            "feature_weights": {
                "post_count": 1.0,
                "interaction_count": 1.0,
                "post_growth_30d": 1.0,
                "interaction_growth_30d": 1.0,
            },
        },
    }

    result = apply_external_media_correction(
        comparable_summary=summary,
        candidates=candidates,
        targets=targets,
        standard_ip_mapping=mapping,
        media_history=media,
        config=config,
    )

    coefficient = result.summary.loc[0, "external_direction_coefficient"]
    assert 1.0 < coefficient <= 1.3
    assert result.summary.loc[0, "external_status"].startswith("applied_")
    assert "anchor_velocity" not in result.diagnostics.columns


def test_external_media_missing_name_match_is_neutral_not_zero() -> None:
    result = apply_external_media_correction(
        comparable_summary=pd.DataFrame([{"sku_code": "T"}]),
        candidates=pd.DataFrame(
            [{
                "target_sku_code": "T", "candidate_sku_code": "C",
                "selected": 1, "alpha_weight": 1.0,
            }]
        ),
        targets=pd.DataFrame(
            [{"sku_code": "T", "first_actual_sale_date": "2026-02-01"}]
        ),
        standard_ip_mapping=pd.DataFrame(
            [
                {"sku_code": "T", "standard_ip_name": "A", "standard_ip_level1": "A", "standard_ip_level2": "", "path1_eligible": 1},
                {"sku_code": "C", "standard_ip_name": "B", "standard_ip_level1": "B", "standard_ip_level2": "", "path1_eligible": 1},
            ]
        ),
        media_history=pd.DataFrame([_social_row("2026-01-31", "另一个IP", 1, 1)]),
        config={
            "external_coefficient_bounds": [0.7, 1.3],
            "external_media": {
                "prelaunch_days": 30,
                "sentiment_coverage_min": 0.5,
                "minimum_snapshot_days_for_trend": 2,
                "coefficient_slope": 0.15,
                "score_difference_bounds": [-2.0, 2.0],
                "feature_weights": {"post_count": 1.0},
            },
        },
    )
    assert result.summary.loc[0, "external_direction_coefficient"] == 1.0
    assert result.summary.loc[0, "external_status"] == "neutral_target_ip_not_matched_to_social_ip_name"


def test_external_media_sums_daily_volume_in_prelaunch_window() -> None:
    result = apply_external_media_correction(
        comparable_summary=pd.DataFrame([{"sku_code": "T"}]),
        candidates=pd.DataFrame(
            [{
                "target_sku_code": "T", "candidate_sku_code": "C",
                "selected": 1, "alpha_weight": 1.0,
            }]
        ),
        targets=pd.DataFrame(
            [{"sku_code": "T", "first_actual_sale_date": "2026-02-01"}]
        ),
        standard_ip_mapping=pd.DataFrame(
            [
                {"sku_code": "T", "standard_ip_name": "目标IP", "standard_ip_level1": "目标IP", "standard_ip_level2": "", "path1_eligible": 1},
                {"sku_code": "C", "standard_ip_name": "IP_B", "standard_ip_level1": "IP_B", "standard_ip_level2": "", "path1_eligible": 1},
            ]
        ),
        media_history=pd.DataFrame(
            [
                _social_row("2026-01-30", "目标IP", 5, 50),
                _social_row("2026-01-31", "目标IP", 0, 0),
                _social_row("2026-01-30", "IP_B", 1, 10),
                _social_row("2026-01-31", "IP_B", 1, 10),
            ]
        ),
        config={
            "external_coefficient_bounds": [0.7, 1.3],
            "external_media": {
                "prelaunch_days": 30,
                "sentiment_coverage_min": 0.5,
                "minimum_snapshot_days_for_trend": 2,
                "coefficient_slope": 0.15,
                "score_difference_bounds": [-2.0, 2.0],
                "feature_weights": {"post_count": 1.0},
            },
        },
    )

    diagnostic = result.diagnostics.iloc[0]
    assert diagnostic["target_media__post_count"] == 5.0
    assert diagnostic["benchmark_media__post_count"] == 2.0
    assert diagnostic["external_direction_coefficient"] > 1.0


def test_external_media_historical_checkpoint_excludes_future_prelaunch_days() -> None:
    result = apply_external_media_correction(
        comparable_summary=pd.DataFrame([{"sku_code": "T"}]),
        candidates=pd.DataFrame(
            [{
                "target_sku_code": "T", "candidate_sku_code": "C",
                "selected": 1, "alpha_weight": 1.0,
            }]
        ),
        targets=pd.DataFrame(
            [{"sku_code": "T", "first_actual_sale_date": "2026-06-01"}]
        ),
        standard_ip_mapping=pd.DataFrame(
            [
                {"sku_code": "T", "standard_ip_name": "目标IP", "standard_ip_level1": "目标IP", "standard_ip_level2": "", "path1_eligible": 1},
                {"sku_code": "C", "standard_ip_name": "IP_B", "standard_ip_level1": "IP_B", "standard_ip_level2": "", "path1_eligible": 1},
            ]
        ),
        media_history=pd.DataFrame(
            [
                _social_row("2026-01-31", "目标IP", 2, 20),
                _social_row("2026-02-02", "目标IP", 999, 9999),
                _social_row("2026-01-31", "IP_B", 1, 10),
                _social_row("2026-02-02", "IP_B", 1, 10),
            ]
        ),
        config={
            "external_coefficient_bounds": [0.7, 1.3],
            "external_media": {
                "prelaunch_days": 30,
                "sentiment_coverage_min": 0.5,
                "minimum_snapshot_days_for_trend": 2,
                "coefficient_slope": 0.15,
                "score_difference_bounds": [-2.0, 2.0],
                "feature_weights": {"post_count": 1.0},
            },
        },
        decision_cutoff_date=date(2026, 1, 31),
    )

    diagnostic = result.diagnostics.iloc[0]
    assert diagnostic["target_media__post_count"] == 2.0
    assert diagnostic["benchmark_media__post_count"] == 1.0
    assert str(diagnostic["window_end"]) == "2026-01-31"
    assert diagnostic["external_status"].startswith("applied_point_in_time_")


def test_external_media_uses_reviewed_social_alias_csv() -> None:
    result = apply_external_media_correction(
        comparable_summary=pd.DataFrame([{"sku_code": "T"}]),
        candidates=pd.DataFrame(
            [{
                "target_sku_code": "T", "candidate_sku_code": "C",
                "selected": 1, "alpha_weight": 1.0,
            }]
        ),
        targets=pd.DataFrame(
            [{"sku_code": "T", "first_actual_sale_date": "2026-02-01"}]
        ),
        standard_ip_mapping=pd.DataFrame(
            [
                {
                    "sku_code": "T", "standard_ip_code": "IP_A::Character A",
                    "standard_ip_name": "Character A", "standard_ip_level1": "IP_A",
                    "standard_ip_level2": "Character A", "path1_eligible": 1,
                },
                {
                    "sku_code": "C", "standard_ip_code": "B",
                    "standard_ip_name": "IP_B", "standard_ip_level1": "IP_B",
                    "standard_ip_level2": "", "path1_eligible": 1,
                },
            ]
        ),
        social_ip_alias_mapping=pd.DataFrame(
            [{
                "standard_ip_code": "IP_A::Character A",
                "social_ip_name": "IP_A_Character A",
                "match_method": "reviewed_exact_alias",
                "enabled": 1,
            }]
        ),
        media_history=pd.DataFrame(
            [
                _social_row("2026-01-31", "IP_A_Character A", 20, 200),
                _social_row("2026-01-31", "IP_B", 1, 10),
            ]
        ),
        config={
            "external_coefficient_bounds": [0.7, 1.3],
            "external_media": {
                "prelaunch_days": 30,
                "sentiment_coverage_min": 0.5,
                "minimum_snapshot_days_for_trend": 2,
                "coefficient_slope": 0.15,
                "score_difference_bounds": [-2.0, 2.0],
                "feature_weights": {"post_count": 1.0},
            },
        },
    )

    diagnostic = result.diagnostics.iloc[0]
    assert result.summary.loc[0, "external_status"].startswith("applied_")
    assert diagnostic["target_external_ip_names"] == "IP_Ahellokitty"
    assert diagnostic["target_external_match_sources"] == (
        "configured_csv:reviewed_exact_alias"
    )


def test_social_alias_loader_requires_shared_bucket_marker(tmp_path) -> None:
    path = tmp_path / "aliases.csv"
    path.write_text(
        "mapping_version,catalog_as_of,standard_ip_code,standard_ip_name,"
        "social_ip_name,match_method,shared_social_bucket,enabled,source_note\n"
        "v1,2026-04-01,A,A,合并IP,reviewed_exact_alias,0,1,test\n"
        "v1,2026-04-01,B,B,合并IP,reviewed_exact_alias,0,1,test\n",
        encoding="utf-8",
    )

    try:
        load_social_ip_alias_mapping(path)
    except DataContractViolation as exc:
        assert "shared_social_bucket" in str(exc)
    else:
        raise AssertionError("expected shared-bucket validation failure")
