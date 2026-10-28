"""V3.2 cohort split, rolling OOF, model fitting, and Test replay."""
from training.modeling import (
    FittedModel,
    TrainingResult,
    fit_model,
    predict,
    replay_test,
    rolling_oof_train,
    weighted_metrics,
)
from training.first_order_labels import (
    FirstOrderLabelResult,
    actual_sales_weekly_from_raw,
    aggregate_weekly_label_inputs,
    build_first_order_wcover_labels,
    build_wcover_windows,
    combine_weekly_label_inputs,
)
from training.section84 import (
    DemandResidualModel,
    FixedDataset,
    Section84TrainingResult,
    fix_section84_dataset,
    fit_demand_residual_model,
    predict_demand_residual,
    replay_section84_locked_test,
    train_section84_first_order,
)
from training.comparable_tuning import (
    ComparableParameterModel,
    apply_comparable_parameters,
    fit_comparable_parameters,
)

__all__ = [
    "FittedModel",
    "TrainingResult",
    "fit_model",
    "predict",
    "replay_test",
    "rolling_oof_train",
    "weighted_metrics",
    "FirstOrderLabelResult",
    "actual_sales_weekly_from_raw",
    "aggregate_weekly_label_inputs",
    "build_first_order_wcover_labels",
    "build_wcover_windows",
    "combine_weekly_label_inputs",
    "DemandResidualModel",
    "FixedDataset",
    "Section84TrainingResult",
    "fix_section84_dataset",
    "fit_demand_residual_model",
    "predict_demand_residual",
    "replay_section84_locked_test",
    "train_section84_first_order",
    "ComparableParameterModel",
    "apply_comparable_parameters",
    "fit_comparable_parameters",
]
