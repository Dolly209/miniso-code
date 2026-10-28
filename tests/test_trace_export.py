from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from data.trace_export import TraceWriter, subset_entities, export_label_weeks
from first_order.pipeline import full_first_order, FullChainReplay, BASE, PREFIX
from training.comparable_tuning import ComparableParameterModel, select_comparable_parameters
from training.section84 import fit_demand_residual_model, DemandResidualModel
from test_step5_full_chain import context_fixture, transform_for


def test_audit_outputs_do_not_change_production_result_and_include_factor_breakdown():
    context=context_fixture()
    chosen=select_comparable_parameters(context["frames"]["comparables"],ComparableParameterModel(("similarity_x",),(1.,),.15,2,.1))
    baseline,_,_=full_first_order(context,chosen,frozen_transform=transform_for(context))
    audit={}
    actual,_,_=full_first_order(context,chosen,frozen_transform=transform_for(context),audit_outputs=audit)
    pd.testing.assert_frame_equal(baseline.decisions,actual.decisions)
    pd.testing.assert_frame_equal(baseline.weekly_demand,actual.weekly_demand)
    assert {"factor_detail","transformed_features","demand_adjustments","effective_factor_config"}<=set(audit)


def test_scoped_export_keeps_related_comparables_and_full_store_universe():
    f=pd.DataFrame(dict(target_sku_code=["T","OTHER"],candidate_sku_code=["C1","C2"]))
    assert subset_entities(f,target="T",related={"T","C1"},name="comparables").candidate_sku_code.tolist()==["C1"]
    f=pd.DataFrame(dict(sku_code=["T","C1","OTHER"],value=[1,2,3]))
    assert subset_entities(f,target="T",related={"T","C1"},name="features").sku_code.tolist()==["T","C1"]
    f=pd.DataFrame(dict(store_code=["S1","S2"],selected=[1,0]))
    assert len(subset_entities(f,target="T",related={"T"},name="stores"))==2
    f=pd.DataFrame(dict(external_ip_key=["wanted","unrelated"],value=[1,2]))
    assert subset_entities(f,target="T",related={"T"},name="social",social_keys={"wanted"}).value.tolist()==[1]


def test_streamed_label_original_rows_and_week_reconciliation_do_not_fill_missing_days(tmp_path):
    source=tmp_path/"parts"
    source.mkdir()
    pd.DataFrame(dict(sku_code=["T","OTHER"],store_code=["S1","S3"],lifecycle_week=[1,1],restored_natural_sales=[4.,999.],clean_sales_qty=[2.,999.])).to_parquet(source/"a.parquet",index=False)
    pd.DataFrame(dict(sku_code=["T","T"],store_code=["S2","S1"],lifecycle_week=[1,2],restored_natural_sales=[6.,500.],clean_sales_qty=[5.,400.])).to_parquet(source/"b.parquet",index=False)
    labels=pd.DataFrame(dict(sku_code=["T"],W_cover=[1],D_first_ref=[10.]))
    writer=TraceWriter(tmp_path/"out")
    report=export_label_weeks(writer,weekly_path=source,labels=labels,skus=["T"])
    assert report["all_label_sums_match"] and report["store_week_rows"]=={"T":3}
    raw=pd.read_csv(writer.root/"label_source/T/store_weekly_original.csv.gz")
    assert len(raw)==3 and raw.audit_in_W_cover.tolist()==[True,True,False]
    assert raw.restored_natural_sales.sum()==510.
    assert "restored_lost_sales" not in raw  # missing data never fabricated
    reconcile=pd.read_csv(writer.root/"label_source/label_reconciliation.csv")
    assert reconcile.iloc[0].reconstructed_from_original_weeks==10.
    assert writer.finish(report).is_file()
    with pytest.raises(ValueError,match="already exists"):TraceWriter(writer.root)


def test_frozen_model_detailed_export_matches_saved_base_prediction_and_store_week_sums(tmp_path,monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from scripts.export_first_order_trace import export_reconstruction
    context=context_fixture()
    transform=transform_for(context)
    comp=ComparableParameterModel(("similarity_x",),(1.,),.15,2,.1)
    replay=FullChainReplay(lambda key:context,key_column="prediction_checkpoint_key")
    row=pd.DataFrame({"prediction_checkpoint_key":["T@M3"],"sku_code":["T"],"split":["train"],
                      "first_actual_sale_date":["2025-04-01"],"label":[100.],BASE:[1.]})
    computed=replay.recompute(row,comp,transform)
    column=PREFIX+"delta_factor__form_play_fit"
    model=fit_demand_residual_model(computed,feature_columns=[column],feature_signs={column:"nonnegative"},
        feature_groups={column:"PRODUCT"},base_demand_column=BASE,ridge=.1,l1_ratio=0.,half_life=52,
        demand_bounds=(.5,2.),coefficient_bounds=context["parameters"]["factors"]["coefficient_bounds"])
    model=DemandResidualModel.from_dict({**model.as_dict(),"frozen_transform":transform})
    result,_,_=full_first_order(context,select_comparable_parameters(context["frames"]["comparables"],comp),
        frozen_transform=transform,learned_model=model,include_details=False)
    expected=pd.Series({"prediction":float(result.decisions.iloc[0].D_demand_adjusted),"base_demand":float(computed.iloc[0][BASE])})
    before=json.dumps(model.as_dict(),sort_keys=True)
    writer=TraceWriter(tmp_path/"export")
    check=export_reconstruction(writer,root="T/OOF",sku="T",key="T@M3",context=context,comparable_model=comp,demand_model=model,expected=expected)
    assert check["base_matches"] and check["prediction_matches"] and check["store_base_matches"] and check["week_base_matches"]
    assert json.dumps(model.as_dict(),sort_keys=True)==before
    assert (writer.root/"T/OOF/08_factor_detail.csv").is_file()
    bad=expected.copy();bad["prediction"]+=100.
    check=export_reconstruction(writer,root="T/final",sku="T",key="T@M3",context=context,comparable_model=comp,demand_model=model,expected=bad)
    assert not check["prediction_matches"]


def test_export_cli_reads_saved_bundles_and_produces_archive_without_any_fit(tmp_path,monkeypatch):
    from pathlib import Path
    from data.artifacts import StageArtifactStore as Store
    from data.recompute import write_recompute_frames, seal_recompute_bundle
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from scripts.export_first_order_trace import main
    context=context_fixture()
    transform=transform_for(context)
    comp=ComparableParameterModel(("similarity_x",),(1.,),.15,2,.1)
    row=pd.DataFrame({"prediction_checkpoint_key":["T@M3"],"sku_code":["T"],"split":["Test"],
                      "first_actual_sale_date":["2025-04-01"],"decision_cutoff_time":[context["cutoff"]],"label":[100.],BASE:[1.]})
    computed=FullChainReplay(lambda key:context,key_column="prediction_checkpoint_key").recompute(row,comp,transform)
    column=PREFIX+"delta_factor__form_play_fit"
    model=fit_demand_residual_model(computed,feature_columns=[column],feature_signs={column:"nonnegative"},
        feature_groups={column:"PRODUCT"},base_demand_column=BASE,ridge=.1,l1_ratio=0.,half_life=52,
        demand_bounds=(.5,2.),coefficient_bounds=context["parameters"]["factors"]["coefficient_bounds"])
    model=DemandResidualModel.from_dict({**model.as_dict(),"frozen_transform":transform})
    result,_,_=full_first_order(context,select_comparable_parameters(context["frames"]["comparables"],comp),
        frozen_transform=transform,learned_model=model,include_details=False)
    saved=computed.assign(prediction=result.decisions.iloc[0].D_demand_adjusted)
    for relative,frame in (("training/fixed_dataset.csv",row),("test/test_predictions.csv",saved),
                           ("training/oof_predictions.csv",saved.iloc[:0]),("training/fold_models.csv",pd.DataFrame(columns=["fold"])),
                           ("training/fit_exclusions.csv",pd.DataFrame(columns=["sku_code"])),
                           ("training/fold_plan.csv",pd.DataFrame(columns=["fold"])),
                           ("training/time_blocks.csv",pd.DataFrame(columns=["sku_code"])),
                           ("training/train_ablation.csv",pd.DataFrame(columns=["stage"])),
                           ("training/model_selection.csv",pd.DataFrame(columns=["ridge"]))):
        Store.write_frame(tmp_path/relative,frame)
    payload={**model.as_dict(),"pipeline_parameters":context["parameters"],"comparable_model":comp.as_dict()}
    for relative,data in (("training/model.json",payload),("training/training_report.json",{}),
                          ("training/comparable_model.json",comp.as_dict()),("test/backtest_report.json",{})):
        Store.write_json(tmp_path/relative,data)
    for role in ("oof","final"):
        group=tmp_path/role/"checkpoint_groups/g"
        Store.write_frame(group/"target_schedule.csv",row)
        write_recompute_frames(group/"round6/_recompute_inputs",context["frames"])
        seal_recompute_bundle(group/"round6/_recompute_inputs",context["parameters"],context["cutoff"])
    Store.write_frame(tmp_path/"final/selected_replay_schedule.csv",row.assign(sku_name="Toy target"))
    Store.write_frame(tmp_path/"labels/first_order_wcover_labels.csv",pd.DataFrame(dict(sku_code=["T"],W_cover=[10],D_first_ref=[100.])))
    Store.write_frame(tmp_path/"label_weekly.csv",pd.DataFrame(dict(sku_code=["T"],store_code=["S"],lifecycle_week=[1],restored_natural_sales=[100.])))
    Store.write_json(tmp_path/"labels/first_order_label_report.json",{"restore_weekly":str(tmp_path/"label_weekly.csv")})
    def no_fit(*a,**kw):raise AssertionError("export must never fit model parameters")
    monkeypatch.setattr("training.section84.fit_demand_residual_model",no_fit)
    monkeypatch.setattr("training.comparable_tuning.fit_comparable_parameters",no_fit)
    argv=["export_first_order_trace.py","--training-dir",str(tmp_path/"training"),"--test-dir",str(tmp_path/"test"),
          "--oof-replay-dir",str(tmp_path/"oof"),"--final-replay-dir",str(tmp_path/"final"),
          "--label-dir",str(tmp_path/"labels"),"--all-fixed-skus","--output-dir",str(tmp_path/"export")]
    monkeypatch.setattr("sys.argv",argv)
    assert main()==0
    report=Store.read_json(tmp_path/"export/trace_report.json")
    assert not report["parameter_fitting_performed"] and not report["errors"]
    assert report["label_source_export"]["all_label_sums_match"]
    assert report["reproduction_checks"][0]["prediction_matches"]
    assert (tmp_path/"export.zip").is_file()
