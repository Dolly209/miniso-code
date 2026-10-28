#!/usr/bin/env python3
"""Bounded Step4 diagnostic. Freeze Top3/alpha, v_model and M5; never train."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from math import isclose
from pathlib import Path
import subprocess
import sys
import zipfile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts._pipeline_common import parse_cutoff, positive_int
from common.logging_utils import setup_run_logger
from config.settings import AppConfig
from data.artifacts import StageArtifactStore as Store
from data.restoration_audit import source_quantity_contract
from restore.step4_audit import exposure_audit, lifecycle_contributions

def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--baseline-replay-dir", type=Path, action="append", required=True,
                   help="Repeat in priority order; latest baseline wins per SKU. Baselines are never modified.")
    p.add_argument("--probe-dir", type=Path, default=Path("artifacts/server_probe_round2"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sku-code", action="append", default=[])
    p.add_argument("--limit-skus", type=positive_int)
    p.add_argument("--batch-skus", type=positive_int, default=5)
    p.add_argument("--decision-cutoff-time", type=parse_cutoff, required=True,
                   help="Dataset upper bound, NOT a replacement for each target's historical cutoff.")
    p.add_argument("--algorithm-config", type=Path, default=Path("config/algorithm_v3_2.json"))
    p.add_argument("--shared-source-cache-dir", type=Path)
    p.add_argument("--arrival-plan-file", type=Path)
    p.add_argument("--constraints-file", type=Path)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def discover_groups(baselines, skus, bound):
    selected = {}
    for baseline in baselines:
        for schedule_path in sorted((absolute(baseline) / "checkpoint_groups").glob("*/target_schedule.csv")):
            schedule = Store.read_frame(schedule_path)
            for sku in schedule.sku_code.astype(str):
                if sku not in skus or sku in selected:
                    continue
                group = schedule_path.parent
                meta = Store.read_json(group / "replay_manifest.json")
                cutoff = parse_cutoff(meta["decision_cutoff_time"])
                if cutoff > bound:
                    raise ValueError(f"{sku}: historical cutoff exceeds dataset bound")
                if parse_cutoff(meta["product_snapshot_time"]) > bound:
                    raise ValueError(f"{sku}: static product snapshot exceeds the confirmed dataset bound")
                cohort = group / "round5_preselection/selected_candidate_cohort.csv"
                candidates = Store.read_frame(cohort)
                if candidates.empty:
                    raise ValueError(f"{sku}: static-only group has no M2 candidates; audit separately, do not fabricate history")
                model = absolute(meta["restoration_model_file"])
                required = [model, group / "round5/candidate_detail.csv", group / "round5/comparable_summary.csv",
                            group / "round7/demand_adjustments.csv", group / "round8/first_order_decisions.csv"]
                for path in required:
                    if not path.is_file():
                        raise FileNotFoundError(f"Step4 prerequisite missing: {path}")
                selected[sku] = (group, meta, cohort, model)
    missing = sorted(set(skus) - set(selected))
    if missing:
        raise ValueError(f"No baseline checkpoint for requested SKUs: {missing}")
    return selected


def commands(args, sku, baseline, meta, cohort, model):
    out = absolute(args.output_dir) / "checkpoint_groups" / sku
    common = ["--data-root", str(args.data_root), "--decision-cutoff-time", meta["decision_cutoff_time"],
              "--product-snapshot-time", meta["product_snapshot_time"]]
    candidates = Store.read_frame(cohort)
    restore = [sys.executable, str(PROJECT_ROOT / "scripts/run_restore.py"), *common,
               "--probe-dir", str(absolute(args.probe_dir)), "--cohort-file", str(cohort),
               "--restoration-model-file", str(model), "--model-application-scope", meta.get("restoration_model_usage_scope", "locked_test"),
               "--batch-skus", str(args.batch_skus), "--skip-daily-exposure-audit", "--skip-partition-parts",
               "--output-dir", str(out / "round4")]
    for code in candidates.sku_code.astype(str).drop_duplicates():
        restore += ["--sku-code", code]
    if args.shared_source_cache_dir:
        restore += ["--shared-source-cache-dir", str(absolute(args.shared_source_cache_dir))]
    placement = [sys.executable, str(PROJECT_ROOT / "scripts/run_placement.py"), *common,
                 "--comparable-dir", str(baseline / "round5"), "--restore-dir", str(out / "round4"),
                 "--sku-code", sku, "--algorithm-config", str(absolute(args.algorithm_config)),
                 "--output-dir", str(out / "round6")]
    if "launch_universe_version" in candidates:
        placement += ["--launch-cohort-file", str(cohort)]
    first_order = [sys.executable, str(PROJECT_ROOT / "scripts/run_first_order.py"), *common,
                   "--comparable-dir", str(baseline / "round5"), "--factor-dir", str(baseline / "round7"),
                   "--placement-dir", str(out / "round6"), "--sku-code", sku,
                   "--algorithm-config", str(absolute(args.algorithm_config)), "--output-dir", str(out / "round8")]
    for flag, path in (("--arrival-plan-file", args.arrival_plan_file), ("--constraints-file", args.constraints_file)):
        if path:
            first_order += [flag, str(absolute(path))]
    return out, [("M2_frozen_apply", restore), ("M3_M4_placement_lifecycle", placement), ("M6_M7_fixed_factor_demand", first_order)]


def main():
    args = parse_args()
    skus = list(dict.fromkeys(args.sku_code))
    if not skus:
        raise ValueError("supply one or more --sku-code values")
    if args.limit_skus:
        skus = skus[:args.limit_skus]
    groups = discover_groups(args.baseline_replay_dir, skus, args.decision_cutoff_time)
    prepared = {sku: commands(args, sku, *group) for sku, group in groups.items()}
    if args.dry_run:
        print("DRY RUN: no writes, subprocesses, raw-data reads or model training")
        for sku, (_, stages) in prepared.items():
            print(f"SKU={sku} historical_cutoff={groups[sku][1]['decision_cutoff_time']}")
            for stage, cmd in stages:
                print(stage + ": " + subprocess.list2cmdline(cmd))
        return 0
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("verify_restoration_lifecycle", config.log_dir, config.timezone)
    root = absolute(args.output_dir)
    if any(root.resolve() == absolute(b).resolve() or root.resolve() in absolute(b).resolve().parents or absolute(b).resolve() in root.resolve().parents for b in args.baseline_replay_dir):
        raise ValueError("Step4 output must be separate from all baseline directories")
    review = root / "review"
    sources = {str(absolute(args.algorithm_config)): digest(absolute(args.algorithm_config))}
    for sku, (baseline, _, cohort, model) in groups.items():
        for path in (cohort, model, baseline / "round5/candidate_detail.csv", baseline / "round7/demand_adjustments.csv"):
            if str(path) not in sources:
                sources[str(path)] = digest(path)
    code_paths = [Path(__file__), *[PROJECT_ROOT / p for p in (
        "src/restore/historical.py", "src/restore/exposure_calendar.py", "src/restore/step4_audit.py",
        "src/restore/batched.py", "src/placement/engine.py", "src/placement/lifecycle.py",
        "src/data/decision_adapter.py", "src/data/restoration_audit.py", "src/first_order/engine.py",
        "src/data/restoration_adapter.py",
        "scripts/run_placement.py", "scripts/run_first_order.py")]]
    for optional in (args.arrival_plan_file, args.constraints_file):
        if optional is not None:
            sources[str(absolute(optional))] = digest(absolute(optional))
    contract = {"skus": skus, "data_root": str(args.data_root.resolve()), "sources": sources,
                "code_sha256": {str(p.relative_to(PROJECT_ROOT)): digest(p) for p in code_paths},
                "commands": {sku: stages for sku, (_, stages) in prepared.items()}}
    if (root / "run_contract.json").is_file() and Store.read_json(root / "run_contract.json") != json.loads(json.dumps(contract)):
        raise ValueError("Step4 inputs/code changed: use a NEW output directory, never overwrite the diagnostic baseline")
    Store.write_json(root / "run_contract.json", contract)
    comparisons, exposure_summaries, contributions, clocks, ledgers, upstream, candidate_clocks = [], [], [], [], [], [], []
    weekly_reconciliation = []
    Store.write_frame(review / "source_quantity_contract.csv", pd.DataFrame(source_quantity_contract()))
    for index, sku in enumerate(skus, 1):
        out, stages = prepared[sku]
        baseline, meta, _, _ = groups[sku]
        for stage, cmd in stages:
            logger.info("step4_progress target=%s targets=%s sku=%s stage=%s event=start", index, len(skus), sku, stage)
            try:
                subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            except subprocess.CalledProcessError as exc:
                logger.exception("step4 failed sku=%s stage=%s", sku, stage)
                Store.write_json(review / "step4_failure.json", {"status": "failed", "sku_code": sku, "stage": stage, "returncode": exc.returncode, "log": str(log_path)})
                return 1
            logger.info("step4_progress target=%s targets=%s sku=%s stage=%s event=complete", index, len(skus), sku, stage)
        weekly = Store.read_frame(out / "round4/restored_store_weekly.csv")
        ledger, summary = exposure_audit(weekly)
        ledger["target_sku_code"] = sku
        summary["target_sku_code"] = sku
        ledgers.append(ledger)
        exposure_summaries.append(summary)
        curve = Store.read_frame(out / "round8/first_order_weekly_demand.csv")
        contributions.append(lifecycle_contributions(curve))
        plan = Store.read_frame(out / "round6/placement_plan.csv")
        clockcols = [c for c in ("sku_code", "store_tier", "tier_week1_anchor", "tier_anchor_status", "v_anchor_source") if c in plan]
        clocks.append(plan[clockcols].drop_duplicates())
        clock_audit = Store.read_frame(out / "round6/restoration_clock_audit.csv")
        clock_audit["target_sku_code"] = sku
        candidate_clocks.append(clock_audit)
        before = Store.read_frame(baseline / "round8/first_order_decisions.csv").set_index("sku_code").loc[sku]
        after = Store.read_frame(out / "round8/first_order_decisions.csv").set_index("sku_code").loc[sku]
        weekly_reconciliation.append(isclose(float(curve.weekly_sales_demand.sum()), float(after.D_sell), rel_tol=1e-9, abs_tol=1e-6))
        row = {"sku_code": sku, "baseline_group": str(baseline), "cutoff": meta["decision_cutoff_time"],
               "arrival_plan_status": after.get("arrival_plan_status"), "comparison_scope": "upstream_diagnostic_not_retrained_model_or_single_factor_ablation"}
        for field in ("D_sell", "D_demand_base", "D_demand_adjusted", "SS_first", "Q_base", "Q_feas", "cover_weeks"):
            row["before_" + field] = before.get(field)
            row["after_" + field] = after.get(field)
        comparisons.append(row)
        report = Store.read_json(out / "round4/round4_report.json")
        upstream.append({"target_sku_code": sku, "model_id": meta.get("restoration_model_id"),
                         "model_gate": report.get("data_quality", {}).get("model_gate", {}),
                         "warnings": report.get("data_quality", {}).get("warnings", [])})
    tables = {"quantity_comparison.csv": pd.DataFrame(comparisons),
              "exposure_summary.csv": pd.concat(exposure_summaries, ignore_index=True),
              "lifecycle_contribution.csv": pd.concat(contributions, ignore_index=True),
              "candidate_lifecycle_clocks.csv": pd.concat(candidate_clocks, ignore_index=True),
              "week1_anchor_sources.csv": pd.concat(clocks, ignore_index=True)}
    for name, table in tables.items():
        Store.write_frame(review / name, table)
    Store.write_frame(root / "velocity_ledger.csv", pd.concat(ledgers, ignore_index=True))
    summary = tables["exposure_summary.csv"]
    checks = {"velocity_formula": bool(summary.max_velocity_formula_error.fillna(0).le(1e-6).all()),
              "repair_denominator": bool(summary.max_denominator_error.le(1e-6).all()),
              "repair_numerator": bool(summary.max_repair_numerator_error.le(1e-6).all()),
              "weekly_demand_equals_D_sell": all(weekly_reconciliation),
              "unknown_not_model_observation": bool(summary.unknown_only_velocity_rows.eq(0).all())}
    Store.write_json(review / "step4_report.json", {"status": "step4_computed_requires_review" if all(checks.values()) else "step4_arithmetic_check_failed",
        "checks": checks, "target_count": len(skus), "model_training_performed": False,
        "frozen_inputs": "original Top3/alpha, v_model and M5; historical cutoffs preserved",
        "units": "schema alone cannot prove business equivalent pieces; no speculative conversion",
        "scope": "evidenced comparable stores only; unknown plan means normal-launch default; no actual target coverage as input",
        "label_changed": False, "upstream_model_warnings": upstream,
        "not_completed": ["formal training/inference unification (step5)", "strict untouched-Test acceptance", "business pack-unit certification"],
        "log": str(log_path)})
    with zipfile.ZipFile(root / "step4_review.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(review.iterdir()):
            archive.write(path, "step4_review/" + path.name)
    print(f"review: {root / 'step4_review.zip'}\nlog: {log_path}\nNo model training or baseline overwrite performed.")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
