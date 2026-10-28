from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from data.artifacts import StageArtifactStore
from data.recompute import write_recompute_frames, seal_recompute_bundle, load_recompute_bundle
from first_order.pipeline import FullChainReplay, BASE
from training.comparable_tuning import ComparableParameterModel
from test_step5_full_chain import context_fixture, transform_for


DIMENSIONS = ("emotion", "ip", "form", "category", "price", "pack", "recent")


def static_context():
    context = deepcopy(context_fixture())
    context["frames"]["comparables"] = pd.DataFrame([{
        "target_sku_code": "T", "candidate_sku_code": "__STATIC__:4", "path_code": "PATH4",
        "path_priority": 4, "hard_filter_pass": 1, "exclusion_reason": "",
        "similarity_score": 0., "similarity_completeness": 0., "quality_score": .2,
        "historical_velocity": 1., "selected": 1, "alpha_weight": 1.,
    }])
    return context


def input_rows():
    return pd.DataFrame({"prediction_checkpoint_key": ["T@M3"], "sku_code": ["T"],
                         "decision_cutoff_time": ["2025-01-01T23:59:59+08:00"], "split": ["train"], BASE: [10.]})


def aggregated(context):
    # Match the actual artifact path: concatenation creates optional columns;
    # CSV roundtrip preserves them as null on the static row.
    raw = context["frames"]["comparables"].copy()
    peer = raw.assign(target_sku_code="OTHER", candidate_sku_code="C1", path_code="PATH1",
                      **{"similarity_" + name: .8 for name in DIMENSIONS})
    return pd.concat([raw.assign(prediction_checkpoint_key="T@M3"),
                      peer.assign(prediction_checkpoint_key="OTHER@M3")], ignore_index=True)


def test_path4_csv_null_dimensions_equal_absent_parquet_dimensions_and_can_recompute(tmp_path):
    context = static_context()
    root = tmp_path / "bundle"
    write_recompute_frames(root, context["frames"])
    seal_recompute_bundle(root, context["parameters"], context["cutoff"])
    loaded = {**load_recompute_bundle(root), "target_sku": "T"}
    csv = tmp_path / "comparable_candidate_detail.csv"
    StageArtifactStore.write_frame(csv, aggregated(context))
    detail = StageArtifactStore.read_frame(csv)
    replay = FullChainReplay(lambda key: loaded, key_column="prediction_checkpoint_key")
    replay.validate_inputs(input_rows(), detail)
    model = ComparableParameterModel(tuple("similarity_" + x for x in DIMENSIONS), (1/7,)*7, .15, 3, .1)
    result = replay.recompute(input_rows(), model, transform_for(context))
    assert np.isfinite(result.iloc[0][BASE]) and result.iloc[0][BASE] > 0
    assert result.iloc[0].step5_selected_candidates == "__STATIC__:4"
    assert not any("similarity_" + name in loaded["frames"]["comparables"] for name in DIMENSIONS)


@pytest.mark.parametrize("field,value", [("similarity_ip", .8), ("alpha_weight", .9),
                                         ("historical_velocity", 2.), ("quality_score", .3)])
def test_real_disagreement_still_fails(field, value):
    context = static_context()
    detail = aggregated(context)
    detail.loc[detail.prediction_checkpoint_key.eq("T@M3"), field] = value
    with pytest.raises(ValueError, match="canonical recompute snapshot differ"):
        FullChainReplay(lambda key: context, key_column="prediction_checkpoint_key").validate_inputs(input_rows(), detail)


def test_canonical_only_populated_dimension_not_ignored_and_core_missing_is_error():
    context = static_context()
    detail = context["frames"]["comparables"].assign(prediction_checkpoint_key="T@M3")
    context["frames"]["comparables"]["similarity_ip"] = .8
    replay = FullChainReplay(lambda key: context, key_column="prediction_checkpoint_key")
    with pytest.raises(ValueError, match="canonical recompute snapshot differ"):
        replay.validate_inputs(input_rows(), detail)
    context["frames"]["comparables"] = context["frames"]["comparables"].drop(columns=["similarity_ip", "alpha_weight"])
    with pytest.raises(ValueError, match="canonical recompute snapshot differ"):
        replay.validate_inputs(input_rows(), detail)


def test_internal_candidate_populated_similarity_cannot_be_replaced_by_null():
    context = context_fixture()
    detail = context["frames"]["comparables"].assign(prediction_checkpoint_key="T@M3")
    detail["similarity_x"] = np.nan
    with pytest.raises(ValueError, match="canonical recompute snapshot differ"):
        FullChainReplay(lambda key: context, key_column="prediction_checkpoint_key").validate_inputs(input_rows(), detail)
