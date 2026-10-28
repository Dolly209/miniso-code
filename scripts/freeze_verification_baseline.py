#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the pre-fix code/config/model/result verification baseline."
    )
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--backtest-dir", type=Path, required=True)
    parser.add_argument("--final-replay-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/verification_baseline_20261025"),
    )
    parser.add_argument("--sku-code", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def git_metadata() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "status_short": status.splitlines(),
        "tracked_diff_sha256": sha256(
            run("diff", "--no-ext-diff", "--binary").encode("utf-8")
        ).hexdigest(),
    }


def main() -> int:
    args = parse_args()
    selected = args.sku_code
    if not selected:
        raise ValueError("supply one or more --sku-code values")
    roles: dict[str, str] = {}
    if args.dry_run:
        print("DRY RUN: verification baseline will not be written")
        print(f"SKUs: {','.join(selected)}")
        print(f"output: {absolute(args.output_dir)}")
        return 0

    output = absolute(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"baseline output already exists and is not empty: {output}; "
            "a frozen baseline is immutable"
        )
    snapshot = output / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=False)

    source_dirs = {
        "training": absolute(args.training_dir),
        "backtest": absolute(args.backtest_dir),
        "final_replay": absolute(args.final_replay_dir),
        "labels": absolute(args.label_dir),
    }
    preferred = {
        "training": (
            "model.json",
            "training_report.json",
            "model_selection.csv",
            "comparable_model.json",
        ),
        "backtest": ("test_predictions.csv", "backtest_report.json"),
        "final_replay": (
            "prelaunch_feature_replay_report.json",
            "replay_manifest.csv",
            "first_order_decisions.csv",
            "placement_summary.csv",
            "comparable_candidate_detail.csv",
        ),
        "labels": (
            "first_order_wcover_labels.csv",
            "formal_training_schedule.csv",
            "first_order_label_report.json",
        ),
    }
    copied: list[dict[str, object]] = []
    for scope, directory in source_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"missing {scope} directory: {directory}")
        for name in preferred[scope]:
            source = directory / name
            if not source.is_file():
                continue
            destination = snapshot / f"{scope}__{name}"
            shutil.copy2(source, destination)
            copied.append(
                {
                    "scope": scope,
                    "source": str(source),
                    "snapshot": str(destination.relative_to(output)),
                    "bytes": source.stat().st_size,
                    "sha256": digest(source),
                }
            )

    config_destination = snapshot / "config__algorithm_v3_2.json"
    shutil.copy2(PROJECT_ROOT / "config/algorithm_v3_2.json", config_destination)
    copied.append(
        {
            "scope": "config",
            "source": str(PROJECT_ROOT / "config/algorithm_v3_2.json"),
            "snapshot": str(config_destination.relative_to(output)),
            "bytes": config_destination.stat().st_size,
            "sha256": digest(config_destination),
        }
    )

    code_rows: list[dict[str, object]] = []
    for root_name in ("src", "scripts", "config"):
        for path in sorted((PROJECT_ROOT / root_name).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".csv"}:
                code_rows.append(
                    {
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": digest(path),
                    }
                )
    pd.DataFrame(code_rows).to_csv(output / "code_config_hashes.csv", index=False)
    pd.DataFrame(
        [
            {"sku_code": code, "verification_role": roles.get(code, "manual")}
            for code in selected
        ]
    ).to_csv(output / "verification_skus.csv", index=False)

    prediction_snapshot = snapshot / "backtest__test_predictions.csv"
    if prediction_snapshot.is_file():
        predictions = pd.read_csv(prediction_snapshot, dtype={"sku_code": str})
        predictions.loc[predictions["sku_code"].isin(selected)].to_csv(
            output / "verification_baseline_predictions.csv", index=False
        )

    manifest = {
        "baseline_status": "frozen_pre_correction_baseline",
        "immutable": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_skus": selected,
        "git": git_metadata(),
        "copied_files": copied,
        "code_config_file_count": len(code_rows),
    }
    (output / "baseline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"baseline: {output}")
    print(f"files: {len(copied)}; code/config hashes: {len(code_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
