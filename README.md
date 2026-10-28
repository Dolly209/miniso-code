# MINISO V3.2 forecasting and replenishment toolkit

This repository contains Python components and command-line tools for a staged first-order and reorder forecasting workflow. The pipeline is split into independent stages so each stage can consume versioned outputs from earlier work.

The implementation is based on the project’s V3.2 specification. A complete code path does not by itself certify results against a production data source or establish calibrated business parameters.

## Architecture

- `src/data/` adapts source records into normalized domain data and validates data contracts.
- `src/comparable/`, `src/restore/`, `src/placement/`, `src/factors/`, `src/first_order/`, `src/reorder/`, and `src/training/` contain the forecasting stages and training utilities.
- `scripts/` provides independent command-line entry points for audits, feature generation, training, replay, and exports.
- `config/` stores versioned algorithm settings, source contracts, and reusable templates.
- `tests/` uses synthetic fixtures to exercise core logic without requiring access to external business data.

## Main entry points

| Workflow | Entry point |
| --- | --- |
| Source and environment inspection | `scripts/server_probe.py` |
| Data contract validation | `scripts/validate_dataset.py` |
| Cohort and training data preparation | `scripts/build_training_dataset.py`, `scripts/build_modeling_dataset.py` |
| Historical demand restoration | `scripts/build_historical_restore.py`, `scripts/run_restore.py` |
| Comparable selection | `scripts/run_comparable.py` |
| Placement and demand factors | `scripts/run_placement.py`, `scripts/run_factors.py` |
| First-order forecast | `scripts/run_first_order.py` |
| Model training and replay | `scripts/train.py`, `scripts/backtest.py` |
| Reorder forecast | `scripts/run_reorder.py` |
| Focused audits and trace export | `scripts/audit_*.py`, `scripts/export_first_order_trace.py` |

Use each command’s `--help` option to see its inputs and output options. Runtime data and generated artifacts should remain outside version control.

## Training and application boundary

Restoration-model fitting and application use separate paths. The rolling training workflow creates fold-specific models from the training population; replay applies a selected frozen model. M1 comparable selection does not fit the restoration model on its selected Top-K records.

The training utilities support chronological splits, out-of-fold feature generation, model selection, and a final fit. Evaluation outputs should be interpreted together with their dataset version, cutoff, and data-quality audit.

## Implemented areas

- Comparable selection with prioritized candidate paths and normalized similarity weights.
- Historical demand restoration with point-in-time safeguards and exposure-state handling.
- Placement and lifecycle estimates with explicit missing-input handling.
- Demand-factor normalization and first-order forecast components.
- Chronological training, replay, and reorder decision components.
- Versioned configuration, audit artifacts, and append-only persistence utilities.

Some parameters require calibration on an approved, representative training set. Missing business inputs are surfaced through configuration and audit artifacts rather than silently inferred from unrelated fields.

## Local setup

Use Python 3.10 or the version declared in `environment.yml`. Install the project dependencies in an isolated environment:

```bash
python -m pip install -r requirements.txt
```

Set local data settings through environment variables. Optional database credentials must be supplied through your local environment and must never be committed. Never commit customer data, generated artifacts, or machine-specific paths.

For example, point a read-only inspection at an approved local dataset:

```bash
python scripts/server_probe.py \
  --data-root /path/to/approved/data \
  --output-dir artifacts/local_probe
```

The default output locations are under `artifacts/`, which is ignored by Git. Review the command help and the applicable data contract before running a workflow against a data source.

## Validation

Synthetic tests cover interfaces, formulas, point-in-time behavior, missing values, training isolation, and persistence logic. Run the suite locally with:

```bash
python -m pytest -q
```

Synthetic tests do not validate production data coverage, model calibration, or database deployment. Those require separately approved data and infrastructure.
