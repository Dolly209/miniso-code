#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from _pipeline_common import add_execution_arguments, print_dry_run  # noqa: E402
from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.artifacts import StageArtifactStore  # noqa: E402
from data.decision_adapter import DecisionDataAdapter  # noqa: E402
from data.domain import filter_skus, frame_hash  # noqa: E402
from data.standard_ip import (  # noqa: E402
    build_server_ip_mapping,
    load_standard_ip_alias_config,
    load_standard_ip_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map the eligible cohort SKUs to standard IPs using the reviewed "
            "reference and one server product snapshot."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--cohort-file",
        type=Path,
        default=Path("artifacts/round3_dataset/cohort.csv"),
        help="Round 3 cohort; only these eligible dataset SKUs are mapped.",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=Path("reference/sku_standard_ip_mapping.csv"),
    )
    parser.add_argument(
        "--alias-config",
        type=Path,
        default=Path("config/standard_ip_aliases.json"),
        help="Aliases and hierarchy defaults derived from the Excel 统计 sheet.",
    )
    add_execution_arguments(parser, default_output="artifacts/standard_ip_mapping")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    outputs = [
        "cohort_sku_ip_mapping.csv",
        "cohort_sku_ip_audit.csv",
        "standard_ip_mapping_report.json",
    ]
    if args.dry_run:
        print_dry_run("standard IP mapping", args, outputs)
        return 0

    config = AppConfig.load(PROJECT_ROOT)
    logger, log_path = setup_run_logger(
        "build_standard_ip_mapping", config.log_dir, config.timezone
    )
    try:
        mapping_path = PROJECT_ROOT / args.mapping_file
        reference = load_standard_ip_mapping(mapping_path)
        alias_config_path = PROJECT_ROOT / args.alias_config
        alias_config = load_standard_ip_alias_config(alias_config_path)
        store = StageArtifactStore()
        cohort = store.read_frame(
            PROJECT_ROOT / args.cohort_file,
            required=("sku_code", "first_actual_sale_date", "split"),
        )
        cohort = filter_skus(
            cohort,
            args.sku_code or None,
            limit_skus=args.limit_skus,
        )
        if cohort["sku_code"].duplicated().any():
            raise ValueError("cohort must contain one row per SKU")
        selected_skus = set(cohort["sku_code"].astype(str))

        adapter = DecisionDataAdapter.from_project(
            project_root=PROJECT_ROOT, data_root=args.data_root, config=config
        )
        product_domain = adapter.products(
            args.decision_cutoff_time,
            sku_codes=sorted(selected_skus),
            allow_missing=True,
            progress=lambda event, details: logger.info(
                "product_progress event=%s %s",
                event,
                " ".join(f"{key}={value}" for key, value in details.items()),
            ),
        )
        products = product_domain.products.copy()
        raw_ip_membership = product_domain.ip_membership.copy()
        result = build_server_ip_mapping(
            products=products,
            raw_ip_membership=raw_ip_membership,
            mapping=reference,
            alias_config=alias_config,
        )

        cohort_context_columns = [
            column
            for column in (
                "sku_code",
                "dataset_version",
                "first_actual_sale_date",
                "split",
                "time_block",
                "sales_band",
            )
            if column in cohort.columns
        ]
        cohort_context = cohort[cohort_context_columns].copy()
        audit = cohort_context.merge(
            result.audit,
            on="sku_code",
            how="left",
            validate="one_to_one",
        )
        audit = audit.merge(
            product_domain.snapshot_audit,
            on="sku_code",
            how="left",
            validate="one_to_one",
        )
        missing_product = audit["mapping_status"].isna()
        audit.loc[missing_product, "mapping_status"] = "missing_product_snapshot"
        audit["standard_ip_count"] = (
            audit["standard_ip_count"].fillna(0).astype(int)
        )
        for column in (
            "standard_ip_codes",
            "mapping_methods",
            "text_ip_candidates",
            "raw_proxy_ip_codes",
        ):
            audit[column] = audit[column].fillna("")

        mapping_output = result.mapping.merge(
            audit[["sku_code", "mapping_status"] + [
                column
                for column in cohort_context_columns
                if column != "sku_code"
            ]],
            on="sku_code",
            how="left",
            validate="many_to_one",
        )
        missing_mapping_skus = selected_skus - set(
            mapping_output["sku_code"].astype(str)
        )
        if missing_mapping_skus:
            audit_by_sku = audit.set_index("sku_code")
            placeholders: list[dict[str, object]] = []
            for sku_code in sorted(missing_mapping_skus):
                audit_row = audit_by_sku.loc[sku_code]
                raw_name = audit_row.get("sku_name", "")
                server_sku_name = "" if pd.isna(raw_name) else str(raw_name)
                placeholder = {column: "" for column in mapping_output.columns}
                placeholder.update(
                    {
                        "sku_code": sku_code,
                        "server_sku_name": server_sku_name,
                        "sku_name": server_sku_name,
                        "mapping_method": "server_unmapped",
                        "mapping_confidence": 0.0,
                        "path1_eligible": 0,
                        "mapping_status": str(audit_row["mapping_status"]),
                        "source_sheet": product_domain.source_partition,
                        "source_row_count": (
                            0
                            if audit_row["mapping_status"] == "missing_product_snapshot"
                            else 1
                        ),
                        "mapping_note": (
                            "截止时点商品快照中未找到该cohort SKU。"
                            if audit_row["mapping_status"] == "missing_product_snapshot"
                            else "服务器SKU未能与Excel参考或文字规则可靠对应。"
                        ),
                    }
                )
                for column in cohort_context_columns:
                    if column != "sku_code":
                        placeholder[column] = audit_row[column]
                placeholders.append(placeholder)
            mapping_output = pd.concat(
                [mapping_output, pd.DataFrame(placeholders)],
                ignore_index=True,
                sort=False,
            )
        mapping_output = mapping_output.sort_values(
            ["sku_code", "standard_ip_code", "mapping_method"], kind="stable"
        ).reset_index(drop=True)

        output_dir = PROJECT_ROOT / args.output_dir
        store.write_frame(output_dir / outputs[0], mapping_output)
        store.write_frame(output_dir / outputs[1], audit)

        status_counts = {
            str(key): int(value)
            for key, value in audit["mapping_status"].value_counts().items()
        }
        mapped_mask = audit["standard_ip_count"].gt(0)
        report = {
            "spec_version": "3.2",
            "stage": "standard IP mapping",
            "status": "implemented_output",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_cutoff_time": args.decision_cutoff_time.isoformat(),
            "source_product_partition": product_domain.source_partition,
            "product_snapshot_policy": "latest_per_cohort_sku_on_or_before_cutoff",
            "product_snapshot_status_counts": {
                str(key): int(value)
                for key, value in product_domain.snapshot_audit[
                    "product_snapshot_status"
                ].value_counts().items()
            },
            "cohort_file": str(args.cohort_file),
            "cohort_skus": int(len(cohort)),
            "mapping_file": str(args.mapping_file),
            "mapping_file_sha256": _sha256(mapping_path),
            "alias_config": str(args.alias_config),
            "alias_config_sha256": _sha256(alias_config_path),
            "reference_rows": int(len(reference)),
            "reference_skus": int(reference["sku_code"].nunique()),
            "cohort_skus_found_in_product_snapshot": int(len(products)),
            "missing_product_snapshot_skus": int(missing_product.sum()),
            "mapped_cohort_skus": int(mapped_mask.sum()),
            "mapping_coverage": (
                float(mapped_mask.mean()) if len(audit) else 0.0
            ),
            "mapping_status_counts": status_counts,
            "mapping_output_hash": frame_hash(mapping_output),
            "audit_output_hash": frame_hash(audit),
        }
        store.write_json(output_dir / outputs[2], report)
        logger.info(
            "standard IP mapping complete cohort_skus=%s mapped=%s coverage=%.4f status_counts=%s",
            len(audit),
            int(mapped_mask.sum()),
            report["mapping_coverage"],
            status_counts,
        )
    except Exception as exc:
        log_exception(
            logger,
            module="build_standard_ip_mapping",
            cutoff=args.decision_cutoff_time,
            exc=exc,
        )
        print(f"ERROR: {type(exc).__name__}: {exc}\nlog: {log_path}")
        return 1

    print(
        "status: implemented_output\n"
        f"mapping: {output_dir / outputs[0]}\n"
        f"audit: {output_dir / outputs[1]}\n"
        f"report: {output_dir / outputs[2]}\n"
        f"log: {log_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
