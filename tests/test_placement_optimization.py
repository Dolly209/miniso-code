from datetime import datetime, date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data.decision_adapter import DecisionDataAdapter
from data.placement_checkpoint import PlacementCheckpointStore, placement_fingerprint
from placement import plan_placement
from test_remaining_rounds import _placement_inputs, PARAMETERS


def inputs():
    targets, comps, stores, history, tiers = _placement_inputs()
    targets = pd.concat([targets, targets.assign(sku_code="T2")], ignore_index=True)
    comps = pd.concat([comps, comps.assign(target_sku_code="T2", candidate_sku_code="C2")], ignore_index=True)
    history = pd.concat([history, history.assign(sku_code="C2", v_neutral=2.0)], ignore_index=True)
    tiers = pd.concat([tiers, tiers.assign(sku_code="C2", v_neutral=2.0)], ignore_index=True)
    return dict(targets=targets, comparables=comps, stores=stores, restored_store_weekly=history,
                restored_tier_weekly=tiers, config=PARAMETERS["placement"])


@pytest.mark.parametrize("missing_rows", [[], [dict(store_code="S1", first_store_arrival_time=None)]])
def test_empty_first_launch_uses_disclosed_prior_not_later_restore_proxy(missing_rows):
    args = inputs()
    class Repository:
        max_rows_per_read = 100
        calls = []
        def read_partition_range(self, **kwargs):
            self.calls.append(kwargs["sku_code"])
            rows = missing_rows if kwargs["sku_code"] == "C1" else [dict(store_code="S1", first_store_arrival_time="2025-01-02")]
            return SimpleNamespace(frame=pd.DataFrame(rows), selected_partitions=(date(2025, 1, 2),))
    repo = Repository()
    domain = DecisionDataAdapter(repo).placement_launch_evidence(
        datetime.fromisoformat("2025-02-01T00:00:00+08:00"),
        candidate_products=pd.DataFrame(dict(sku_code=["C1", "C2"], first_sale_date=["2025-01-01"] * 2, release_dt=[None] * 2)),
        eligible_stores=args["stores"], restored_store_weekly=args["restored_store_weekly"], launch_window_days=28,
    )
    assert repo.calls == ["C1", "C2"]
    assert set(domain.evidence.sku_code) == {"C2"}
    assert "C1" in domain.warnings[0]
    result = plan_placement(**args, placement_evidence=domain.evidence)
    evidence = result.compliance.query("concept == 'launch_window_prior_and_performance_probability'").set_index("target_sku_code")
    assert evidence.loc["T", "source"] == "missing_launch_window_evidence_versioned_prior"
    assert evidence.loc["T", "status"] == "VERSIONED_INITIAL_PARAMETER"
    assert evidence.loc["T2", "source"] == "raw_launch_window_empirical_probabilities"
    summary = result.summary.set_index("sku_code")
    assert summary.loc["T", "evidence_supported_store_count"] == 0
    assert summary.loc["T", "placement_rate"] == PARAMETERS["placement"]["initial_defaults"]["missing_launch_evidence_placement_rate"]
    assert set(
        result.store_plan.loc[
            (result.store_plan["sku_code"] == "T")
            & (result.store_plan["selected"] == 1),
            "placement_origin",
        ]
    ) == {"missing_launch_evidence_versioned_prior"}


def test_interruption_resume_and_changed_inputs_invalidate(tmp_path):
    args = inputs()
    fingerprint = placement_fingerprint([args["restored_store_weekly"]], {"version": 1})
    store = PlacementCheckpointStore(tmp_path, fingerprint)
    def interrupt_after_save(sku, result):
        store.save(sku, result)
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        plan_placement(**args, save_target=interrupt_after_save)
    events = []
    resumed = plan_placement(**args, load_target=store.load, save_target=store.save,
                             progress=lambda event, details: events.append(event))
    fresh = plan_placement(**args)
    assert events.count("target_checkpoint_reused") == 1
    for name in store.names:
        pd.testing.assert_frame_equal(getattr(resumed, name), getattr(fresh, name), check_dtype=False, atol=1e-12, rtol=1e-12)
    changed = args["restored_store_weekly"].copy()
    changed.loc[0, "v_neutral"] += 1
    new_key = placement_fingerprint([changed], {"version": 1})
    assert new_key != fingerprint
    assert PlacementCheckpointStore(tmp_path, new_key).load("T") is None
    assert PlacementCheckpointStore(tmp_path, fingerprint, resume=False).load("T") is None
    assert placement_fingerprint([args["restored_store_weekly"]], {"version": 2}) != fingerprint


def test_empirical_probabilities_use_conditional_groups_and_missing_does_not_expand():
    from placement.engine import _candidate_reasonableness
    evidence = pd.DataFrame({
        "sku_code": ["C"] * 4, "store_code": ["01", "02", "03", "04"],
        "store_tier": ["A"] * 4, "eligible_at_launch": [1, 1, 1, 0],
        "arrived_in_launch_window": [1, 1, 0, 1],
        "v_neutral": [5., 1., np.nan, 8.], "stock_pressure": [0., 8., np.nan, 0.],
        "evidence_weight": [2., 1., 1., 3.],
    })
    policy = pd.DataFrame([dict(store_tier="A", v_min=2., inv_max=4., tau_tier=.5)])
    result = _candidate_reasonableness(evidence=evidence, policy=policy, shrinkage_k=4.)
    np.testing.assert_allclose(result.prior_probability, [0.75] * 3)
    np.testing.assert_allclose(result.performance_probability, [2 / 3] * 3)
    np.testing.assert_allclose(result.reliability_r, [3 / 7] * 3)
    np.testing.assert_allclose(result.p_star, [5 / 7] * 3)
    assert result.reasonable.tolist() == [1, 1, 1]
    assert (
        result.evidence_scope.iloc[2]
        == "group_probability_but_nonarrival_insufficient_no_auto_expand"
    )
    assert result.store_code.tolist() == ["01", "02", "03"]


def test_shared_candidate_computed_once(monkeypatch):
    import placement.engine as engine
    args = inputs()
    args["comparables"]["candidate_sku_code"] = "C1"
    original = engine._candidate_reasonableness
    calls = []
    def counted(**kwargs):
        calls.append(1)
        return original(**kwargs)
    monkeypatch.setattr(engine, "_candidate_reasonableness", counted)
    evidence = pd.DataFrame(
        [
            {
                "sku_code": "C1",
                "store_code": str(row.store_code),
                "store_tier": str(row.store_tier),
                "eligible_at_launch": 1,
                "arrived_in_launch_window": int(row.store_code != "S3"),
                "v_neutral": 2.0,
                "stock_pressure": 1.0,
                "evidence_weight": 1.0,
            }
            for row in args["stores"].itertuples(index=False)
        ]
    )
    result = plan_placement(**args, placement_evidence=evidence)
    assert len(calls) == 1
    assert len(result.summary) == 2


def test_partial_launch_evidence_skips_missing_candidate_and_renormalizes_alpha():
    targets, comparables, stores, history, tiers = _placement_inputs()
    comparables = pd.concat(
        [
            comparables.assign(alpha_weight=0.25),
            comparables.assign(candidate_sku_code="C2", alpha_weight=0.75),
        ],
        ignore_index=True,
    )
    history = pd.concat(
        [history, history.assign(sku_code="C2", v_neutral=20.0)],
        ignore_index=True,
    )
    tiers = pd.concat(
        [tiers, tiers.assign(sku_code="C2", v_neutral=20.0)],
        ignore_index=True,
    )
    evidence = pd.DataFrame(
        [
            {
                "sku_code": "C1",
                "store_code": str(row.store_code),
                "store_tier": str(row.store_tier),
                "eligible_at_launch": 1,
                "arrived_in_launch_window": int(row.store_code != "S3"),
                "v_neutral": 2.0,
                "stock_pressure": 1.0,
                "evidence_weight": 1.0,
            }
            for row in stores.itertuples(index=False)
        ]
    )
    partial = plan_placement(
        targets=targets,
        comparables=comparables,
        stores=stores,
        restored_store_weekly=history,
        restored_tier_weekly=tiers,
        config=PARAMETERS["placement"],
        placement_evidence=evidence,
    )
    c1_only = plan_placement(
        targets=targets,
        comparables=comparables.iloc[:1].assign(alpha_weight=1.0),
        stores=stores,
        restored_store_weekly=history,
        restored_tier_weekly=tiers,
        config=PARAMETERS["placement"],
        placement_evidence=evidence,
    )
    assert partial.summary.iloc[0]["placement_rate"] == pytest.approx(
        c1_only.summary.iloc[0]["placement_rate"]
    )
    assert (
        partial.summary.iloc[0]["placement_evidence_source"]
        == "partial_launch_window_evidence_missing_candidates"
    )
    assert partial.summary.iloc[0]["missing_launch_evidence_candidate_count"] == 1


def test_historical_launch_tier_not_currently_eligible_gets_policy_but_not_selection():
    args = inputs()
    args["targets"] = args["targets"].iloc[:1].copy()
    args["comparables"] = args["comparables"].loc[
        args["comparables"]["target_sku_code"].eq("T")
    ].copy()
    historical_closed = pd.DataFrame(
        [
            {
                "sku_code": "C1",
                "store_code": "CLOSED",
                "store_tier": "7|A",
                "eligible_at_launch": 1,
                "arrived_in_launch_window": 1,
                "v_neutral": 2.0,
                "stock_pressure": 3.0,
                "evidence_weight": 1.0,
            }
        ]
    )
    current = pd.DataFrame(
        [
            {
                "sku_code": "C1",
                "store_code": str(row.store_code),
                "store_tier": str(row.store_tier),
                "eligible_at_launch": 1,
                "arrived_in_launch_window": 1,
                "v_neutral": 2.0,
                "stock_pressure": 3.0,
                "evidence_weight": 1.0,
            }
            for row in args["stores"].itertuples(index=False)
        ]
    )
    result = plan_placement(
        **args,
        placement_evidence=pd.concat(
            [current, historical_closed], ignore_index=True
        ),
    )
    assert "CLOSED" not in set(result.store_plan["store_code"].astype(str))
    assert len(result.summary) == 1
