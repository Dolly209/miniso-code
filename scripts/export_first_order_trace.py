#!/usr/bin/env python3
"""Export saved evidence and reproduce selected forecasts with frozen models.

No training, no parameter search, no original sales/inventory/placement scans.
An optional product read uses the existing adapter at the confirmed snapshot.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.artifacts import StageArtifactStore as Store
from data.recompute import ReplayBundleStore
from data.trace_export import DEFAULT_TRACE_SKUS, TraceWriter, canonical_frame, subset_entities, export_label_weeks
from first_order.pipeline import full_first_order
from training.comparable_tuning import ComparableParameterModel, select_comparable_parameters
from training.section84 import DemandResidualModel


def absolute(value):
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-dir", type=Path, required=True)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--oof-replay-dir", type=Path, required=True)
    p.add_argument("--final-replay-dir", type=Path, required=True)
    p.add_argument("--label-dir", type=Path, required=True)
    p.add_argument("--label-restore-weekly", type=Path)
    p.add_argument("--resolved-standard-ip-file", type=Path)
    p.add_argument("--data-root", type=Path, help="Optional: product attributes only; no original fact-table scans.")
    p.add_argument("--product-snapshot-time", default="2026-09-30T23:59:59+08:00")
    p.add_argument("--sku-code", action="append", default=[])
    p.add_argument("--all-fixed-skus", action="store_true", help="Export detailed evidence for all fixed Train/Test targets, not only the default eight diagnostic cases.")
    p.add_argument("--limit-skus", type=int)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def read(root, filename):
    return Store.read_frame(root / filename)


def fold_models(row):
    theta = json.loads(row["comparable_weights_json"])
    comparable = ComparableParameterModel(tuple(theta), tuple(float(v) for v in theta.values()),
        float(row["comparable_temperature"]), int(row["comparable_top_k"]),
        float(row["similarity_regularization"]), "saved_OOF_fold_parameters")
    model = DemandResidualModel.from_dict(json.loads(row["model_json"]))
    return comparable, model


def export_reconstruction(writer, *, root: str, sku: str, key: str, context: dict,
                          comparable_model, demand_model, expected: pd.Series | None):
    detail = context["frames"]["comparables"]
    detail = detail.loc[detail.target_sku_code.astype(str).eq(sku)].copy()
    chosen = select_comparable_parameters(detail, comparable_model)
    captured = {}
    result, placement, factors = full_first_order(context, chosen,
        frozen_transform=demand_model.frozen_transform, learned_model=demand_model,
        include_details=True, audit_outputs=captured)
    for filename, frame in (
        ("01_comparable_envelope.csv", detail), ("02_trained_selected_comparables.csv", chosen),
        ("03_placement_summary.csv", placement.summary), ("04_all_store_selection.csv.gz", placement.store_plan),
        ("05_placement_diagnostics.csv", placement.diagnostics), ("06_compliance.csv", placement.compliance),
        ("07_tier_lifecycle_curve.csv", placement.demand_curve),
        ("08_factor_detail.csv", captured["factor_detail"]),
        ("09_transformed_target_and_peer_features.csv", captured["transformed_features"]),
        ("10_demand_adjustments.csv", captured["demand_adjustments"]),
        ("11_training_factor_kernel.csv", factors),
        ("12_store_quantity_targets.csv.gz", result.store_targets),
        ("13_tier_week_demand.csv", result.weekly_demand),
        ("14_calculation_steps.csv", result.calculation_detail),
        ("15_first_order_decisions.csv", result.decisions), ("16_prediction_horizons.csv", result.predictions),
    ):
        writer.frame(root + "/" + filename, frame, source="frozen canonical bundle + saved trained model", scope="reproduced; never refitted")
    writer.json(root + "/trained_models.json", {"comparable_model": comparable_model.as_dict(),
        "demand_model": demand_model.as_dict(), "effective_factor_config": captured["effective_factor_config"],
        "warnings": list(result.warnings) + list(placement.warnings)})
    value = result.decisions.iloc[0]
    base, prediction = float(value.D_demand_base), float(value.D_demand_adjusted)
    checks = {"sku_code": sku, "prediction_checkpoint_key": key, "scope": root,
              "reproduced_base": base, "reproduced_prediction": prediction, "comparison_available": expected is not None}
    if expected is not None:
        saved_base = expected.get("base_demand", expected.get("feature__first_order__D_demand_base", np.nan))
        checks.update(saved_base=float(saved_base), saved_prediction=float(expected["prediction"]),
                      base_matches=bool(np.isclose(base, float(saved_base), rtol=1e-8, atol=1e-6)),
                      prediction_matches=bool(np.isclose(prediction, float(expected["prediction"]), rtol=1e-8, atol=1e-6)))
        writer.frame(root + "/saved_prediction.csv", expected.to_frame().T, source="training/test output", scope="original saved result")
    checks["sum_store_base"] = float(result.store_targets["pure_demand_base"].sum()) if not result.store_targets.empty else np.nan
    checks["sum_week_sales_base"] = float(result.weekly_demand["weekly_sales_demand"].sum()) if not result.weekly_demand.empty else np.nan
    checks["store_base_matches"] = bool(np.isclose(checks["sum_store_base"], base, rtol=1e-8, atol=1e-6))
    checks["week_base_matches"] = bool(np.isclose(checks["sum_week_sales_base"], base, rtol=1e-8, atol=1e-6))
    writer.json(root + "/reproduction_checks.json", checks)
    return checks


def main():
    args = parse_args()
    if args.all_fixed_skus and args.sku_code:
        raise ValueError("choose --all-fixed-skus or explicit --sku-code, not both")
    skus = list(dict.fromkeys(args.sku_code or DEFAULT_TRACE_SKUS))
    if not args.all_fixed_skus and not skus:
        raise ValueError("supply --sku-code or select --all-fixed-skus")
    if args.limit_skus is not None:
        if args.limit_skus <= 0:
            raise ValueError("limit-skus must be positive")
        skus = skus[:args.limit_skus]
    if args.dry_run:
        print(f"DRY RUN selected={'all fixed Train/Test' if args.all_fixed_skus else skus}; export all fixed SKU attributes; reconstruct frozen models only; no fitting")
        return 0
    training, test = absolute(args.training_dir), absolute(args.test_dir)
    oof_root, final_root = absolute(args.oof_replay_dir), absolute(args.final_replay_dir)
    labels_root = absolute(args.label_dir)
    writer = TraceWriter(absolute(args.output_dir))
    report = {"mode": "read_only_export_and_frozen_model_reproduction", "selected_skus": skus,
              "parameter_fitting_performed": False, "missing_sources": [], "reproduction_checks": [], "errors": [],
              "daily_policy": "previously pruned daily rows are not reconstructed or fabricated",
              "original_vs_reproduced": "original_replay directories are reference-parameter outputs; trained_reproduction uses saved fold/final models"}
    fixed, oof, predictions, folds = (read(training,"fixed_dataset.csv"), read(training,"oof_predictions.csv"),
                                    read(test,"test_predictions.csv"), read(training,"fold_models.csv"))
    if args.all_fixed_skus:
        skus=list(fixed.sku_code.astype(str).drop_duplicates())
        if args.limit_skus is not None:skus=skus[:args.limit_skus]
        report["selected_skus"]=skus
    payload = Store.read_json(training / "model.json")
    parameters = payload["pipeline_parameters"]
    final_demand = DemandResidualModel.from_dict(payload)
    final_comparable = ComparableParameterModel.from_dict(payload["comparable_model"])
    labels = read(labels_root,"first_order_wcover_labels.csv")
    schedule = read(final_root,"selected_replay_schedule.csv")
    if set(skus) - set(fixed.sku_code.astype(str)):
        raise ValueError(f"requested SKU outside fixed pilot: {sorted(set(skus)-set(fixed.sku_code.astype(str)))}")
    oof_store, final_store = ReplayBundleStore(oof_root,parameters), ReplayBundleStore(final_root,parameters)
    for name,frame in (("all_fixed_dataset",fixed), ("all_Train_attributes",schedule.loc[schedule.split.str.lower().eq("train")]),
                       ("all_Test_attributes",schedule.loc[schedule.split.str.lower().eq("test")]),
                       ("all_Train_OOF_predictions",oof), ("all_Test_predictions",predictions), ("fold_models",folds)):
        writer.frame("overview/"+name+".csv",frame,source="original saved pilot/training/test outputs",scope="all fixed pilot rows; all columns retained")
    for filename in ("training_report.json","comparable_model.json","model.json"):
        writer.json("overview/"+filename,Store.read_json(training/filename))
    writer.json("overview/backtest_report.json",Store.read_json(test/"backtest_report.json"))
    for filename in ("fit_exclusions.csv","fold_plan.csv","time_blocks.csv","train_ablation.csv","model_selection.csv"):
        writer.frame("overview/"+filename,read(training,filename),source=str(training/filename),scope="original saved results")
    label_report = Store.read_json(labels_root/"first_order_label_report.json")
    writer.json("label_source/original_label_report.json",label_report)
    selected_labels=labels.loc[labels.sku_code.astype(str).isin(skus)].copy()
    writer.frame("label_source/selected_labels_all_columns.csv",selected_labels,source=str(labels_root),scope="all original label fields")
    properties=[]
    for key,(bundle,sku) in final_store.index.items():
        if sku not in set(fixed.sku_code.astype(str)):
            continue
        features=canonical_frame(bundle,"features")
        if features is not None:
            target=features.loc[features.sku_code.astype(str).eq(sku)].copy()
            target.insert(0,"audit_prediction_checkpoint_key",key)
            properties.append(target)
    if properties:
        all_properties=pd.concat(properties,ignore_index=True)
        all_properties=all_properties.merge(fixed[["sku_code","split"]],on="sku_code",how="left",validate="many_to_one",suffixes=("","_fixed"))
        writer.frame("overview/all_canonical_product_and_factor_features.csv",all_properties,source="final canonical features",scope="all fixed25; raw attributes and untransformed features")
        writer.frame("overview/Train20_canonical_product_and_factor_features.csv",all_properties.loc[all_properties.split.str.lower().eq("train")],source="final canonical features",scope="all Train including blocked and B1 warmup")
    all_related=set(fixed.sku_code.astype(str))
    for sku in skus:
        sample=fixed.loc[fixed.sku_code.astype(str).eq(sku)]
        key=str(sample.iloc[0].prediction_checkpoint_key)
        writer.frame(f"cases/{sku}/sample_and_label.csv",sample,source=str(training),scope="original fixed row; all modeling columns")
        roles=[("final",final_store,final_root)]
        if str(sample.iloc[0].split).lower()=="train":
            roles.insert(0,("OOF",oof_store,oof_root))
        for role,store,replay_root in roles:
            print(f"trace_export sku={sku} role={role} phase=start",flush=True)
            try:
                context=store(key)
                folder=f"cases/{sku}/{role}"
                source_group=store.index[key][0].parent.parent
                for relative in ("replay_manifest.json","round4/round4_report.json","round5/round5_report.json","round6/round6_report.json","round7/factor_report.json"):
                    path=source_group/relative
                    if path.is_file():writer.json(folder+"/original_replay/"+relative,Store.read_json(path))
                candidates=context["frames"]["comparables"]
                related={sku}|set(candidates.loc[candidates.target_sku_code.astype(str).eq(sku),"candidate_sku_code"].astype(str))
                related={x for x in related if not x.startswith("__STATIC__")}
                all_related.update(related)
                aliases=context["frames"].get("social_aliases",pd.DataFrame())
                social_keys=set(aliases.loc[aliases.sku_code.astype(str).isin(related),"external_ip_key"].astype(str)) if "sku_code" in aliases else set()
                for name,frame in context["frames"].items():
                    subset=subset_entities(frame,target=sku,related=related,name=name,social_keys=social_keys)
                    writer.frame(folder+"/canonical_inputs/"+name+".csv.gz",subset,
                        source=str(store.index[key][0]/(name+".parquet")),scope="frozen input; all source columns; target and related candidates only")
                writer.json(folder+"/canonical_inputs/manifest.json",Store.read_json(store.index[key][0]/"manifest.json"))
                for relative in ("target_schedule.csv","round5/candidate_detail.csv","round5/comparable_summary.csv",
                    "round5_preselection/selected_comparables.csv","round5_preselection/selected_candidate_cohort.csv",
                    "round6/placement_plan.csv","round6/placement_summary.csv","round6/demand_curve.csv",
                    "round6/placement_diagnostics.csv","round6/placement_evidence.csv","round6/target_store_features.csv",
                    "round6/restoration_clock_audit.csv","round7/factor_detail.csv","round7/demand_adjustments.csv",
                    "round7/transformed_features.csv","round8/first_order_decisions.csv",
                    "round8/first_order_calculation_detail.csv","round8/first_order_store_targets.csv","round8/first_order_weekly_demand.csv"):
                    path=source_group/relative
                    if not path.is_file():
                        report["missing_sources"].append({"sku_code":sku,"role":role,"source":str(path)})
                        continue
                    frame=Store.read_frame(path)
                    subset=subset_entities(frame,target=sku,related=related,name=relative,social_keys=social_keys)
                    writer.frame(folder+"/original_replay/"+relative+".gz",subset,source=str(path),scope="original reference-parameter replay; not trained predictions")
                saved=None
                if role=="OOF":
                    hit=oof.loc[oof.prediction_checkpoint_key.astype(str).eq(key)]
                    if hit.empty:
                        writer.json(folder+"/trained_reproduction_status.json",{"status":"no_OOF_prediction","reason":"B1 is warmup or row explicitly excluded; no fake OOF model created"})
                        continue
                    saved=hit.iloc[0]
                    comparable,demand=fold_models(folds.loc[pd.to_numeric(folds.fold).eq(int(saved.fold))].iloc[0])
                else:
                    comparable,demand=final_comparable,final_demand
                    hit=predictions.loc[predictions.prediction_checkpoint_key.astype(str).eq(key)]
                    if not hit.empty:saved=hit.iloc[0]
                    blocked=sample.get("feature__first_order__decision_status",pd.Series(dtype=str)).eq("blocked_no_supported_stores").any()
                    if blocked:
                        writer.json(folder+"/trained_reproduction_status.json",{"status":"excluded_Train_no_forecast","reason":"no_evidence_supported_store"})
                        continue
                check=export_reconstruction(writer,root=folder+"/trained_reproduction",sku=sku,key=key,context=context,
                    comparable_model=comparable,demand_model=demand,expected=saved)
                report["reproduction_checks"].append(check)
                if (saved is not None and not(check["base_matches"] and check["prediction_matches"])) or not(check["store_base_matches"] and check["week_base_matches"]):
                    report["errors"].append({"sku_code":sku,"role":role,"error":"frozen reproduction or quantity reconciliation differs; inspect checks; source not overwritten"})
                print(f"trace_export sku={sku} role={role} phase=complete",flush=True)
            except Exception as exc:
                report["errors"].append({"sku_code":sku,"role":role,"error":f"{type(exc).__name__}: {exc}"})
                print(f"trace_export sku={sku} role={role} phase=error error={type(exc).__name__}: {exc}",flush=True)
    if args.resolved_standard_ip_file:
        ip_path=absolute(args.resolved_standard_ip_file)
        if ip_path.is_file():
            writer.frame("overview/standard_IP_mapping_targets_and_peers.csv",Store.read_frame(ip_path,filters={"sku_code":sorted(all_related)},chunksize=100000),source=str(ip_path),scope="existing reviewed mapping; no new name inference")
        else:report["missing_sources"].append({"source":str(ip_path)})
    if args.data_root:
        from config.settings import AppConfig
        from data.decision_adapter import DecisionDataAdapter
        try:
            adapter=DecisionDataAdapter.from_project(project_root=PROJECT_ROOT,data_root=absolute(args.data_root),config=AppConfig.load(PROJECT_ROOT))
            snapshot=datetime.fromisoformat(args.product_snapshot_time)
            if snapshot.date().isoformat()!="2026-09-30":raise ValueError("this fixed pilot export requires the confirmed 2026-09-30 product snapshot")
            domain=adapter.products(snapshot,sku_codes=sorted(all_related),allow_missing=True,fallback_to_historical=False)
            if domain.source_partition!="2026-09-30":raise ValueError(f"confirmed product partition missing; resolved {domain.source_partition} instead")
            for name,frame in (("product_attributes",domain.products),("source_IP_memberships",domain.ip_membership),("product_snapshot_audit",domain.snapshot_audit)):
                writer.frame("overview/master_20260930_"+name+".csv",frame,source="existing DecisionDataAdapter at 2026-09-30 only",scope="diagnostic enrichment; not replacement training inputs")
            writer.frame("overview/Train20_master_attributes.csv",domain.products.loc[domain.products.sku_code.astype(str).isin(set(fixed.loc[fixed.split.str.lower().eq('train'),'sku_code'].astype(str)))],source="2026-09-30 product adapter",scope="all20 Train")
        except Exception as exc:report["missing_sources"].append({"source":"product master at 2026-09-30","reason":f"{type(exc).__name__}: {exc}"})
    weekly=args.label_restore_weekly or label_report.get("restore_weekly")
    path=absolute(weekly) if weekly else None
    if path is not None and not path.exists() and path.with_name(path.stem+"_parts").is_dir():
        path=path.with_name(path.stem+"_parts")
    try:
        report["label_source_export"]=export_label_weeks(writer,weekly_path=path,labels=selected_labels,skus=skus,
            progress=lambda d:print(f"trace_export {d}",flush=True))
    except Exception as exc:
        report["label_source_export"]={"status":"error","source":str(path),"reason":f"{type(exc).__name__}: {exc}"}
        report["errors"].append({"source":"label_source","error":f"{type(exc).__name__}: {exc}"})
    if report["label_source_export"].get("all_label_sums_match") is False:
        report["errors"].append({"source":"label_source","error":"saved label differs from original W_cover weekly sum; see label_reconciliation.csv"})
    report["all_related_sku_codes"]=sorted(all_related)
    report["status"]="exported_with_errors" if report["errors"] else "exported"
    archive=writer.finish(report)
    print(f"status: {report['status']}\nreport: {writer.root/'trace_report.json'}\narchive: {archive}")
    return 2 if report["errors"] else 0


if __name__=="__main__":
    raise SystemExit(main())
