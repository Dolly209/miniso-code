#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import add_execution_arguments, print_dry_run  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.modeling_dataset import assemble_modeling_dataset  # noqa: E402


def _feature_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("feature artifact must be NAMESPACE=PATH")
    namespace, path = value.split("=", 1)
    if not namespace or not path:
        raise argparse.ArgumentTypeError("feature artifact must be NAMESPACE=PATH")
    return namespace, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the standardized ROUND 8 modeling dataset.")
    parser.add_argument("--cohort-file", type=Path, default=Path("artifacts/round3_dataset/cohort.csv"))
    parser.add_argument("--label-file", type=Path, default=None)
    parser.add_argument("--label-column", default="sales_26w_qty")
    parser.add_argument("--feature-artifact", action="append", type=_feature_spec, default=[])
    add_execution_arguments(parser, default_output="artifacts/training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger("build_modeling_dataset", config.log_dir, config.timezone)
    outputs = ["modeling_dataset.csv", "modeling_dataset_report.json"]
    if args.dry_run:
        print_dry_run("modeling dataset assembly", args, outputs)
        return 0
    try:
        artifact = StageArtifactStore()
        cohort = artifact.read_frame(
            PROJECT_ROOT / args.cohort_file,
            sku_codes=args.sku_code, limit_skus=args.limit_skus,
        )
        labels = None if args.label_file is None else artifact.read_frame(PROJECT_ROOT / args.label_file)
        features = [
            (namespace, artifact.read_frame(PROJECT_ROOT / path))
            for namespace, path in args.feature_artifact
        ]
        dataset = assemble_modeling_dataset(
            cohort=cohort, feature_frames=features, label_frame=labels,
            label_column=args.label_column,
        )
        output_dir = PROJECT_ROOT / args.output_dir
        artifact.write_frame(output_dir / outputs[0], dataset)
        artifact.write_json(output_dir / outputs[1], {
            "spec_version": "3.2", "status": "implemented_output",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "rows": len(dataset), "skus": dataset["sku_code"].nunique(),
            "label_source": str(args.label_file or args.cohort_file),
            "label_column": args.label_column,
            "label_warning": (
                "sales_26w_qty is a demand-learning label, not a universal executed first-order quantity"
                if args.label_column == "sales_26w_qty" else "external mature label supplied"
            ),
            "feature_artifacts": {name: str(path) for name, path in args.feature_artifact},
        })
        logger.info("modeling dataset complete rows=%s", len(dataset))
    except Exception as exc:
        log_exception(logger, module="build_modeling_dataset", cutoff=args.decision_cutoff_time, exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1
    print(f"status: implemented_output\ndataset: {output_dir / outputs[0]}\nreport: {output_dir / outputs[1]}\nlog: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
