from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when configuration is absent or invalid."""


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str = "miniso"

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        names = {
            "host": "MINISO_DB_HOST",
            "port": "MINISO_DB_PORT",
            "user": "MINISO_DB_USER",
            "password": "MINISO_DB_PASSWORD",
            "database": "MINISO_DB_NAME",
        }
        missing = [env_name for env_name in names.values() if not os.environ.get(env_name)]
        if missing:
            raise ConfigError(f"Missing database environment variables: {', '.join(missing)}")
        raw_port = os.environ[names["port"]]
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ConfigError("MINISO_DB_PORT must be an integer") from exc
        database = os.environ[names["database"]]
        if database != "miniso":
            raise ConfigError("MINISO_DB_NAME must be 'miniso' for this project")
        user = os.environ[names["user"]]
        if user.strip().lower() == "root":
            raise ConfigError("MINISO_DB_USER must not be root")
        return cls(
            host=os.environ[names["host"]],
            port=port,
            user=user,
            password=os.environ[names["password"]],
            database=database,
        )


@dataclass(frozen=True)
class ProbeConfig:
    table_specs_path: Path
    sample_files_per_table: int = 3
    max_sample_files_to_attempt: int = 50
    sample_rows_per_file: int = 2_000
    max_sample_rows_per_table: int = 5_000
    recent_partition_limit: int = 10
    sample_file_limit: int = 10


@dataclass(frozen=True)
class TrainingConfig:
    train_ratio: float
    reference_cohort_size: int
    reference_train_size: int
    reference_test_size: int
    reference_train_test_cutoff: date
    n_time_blocks: int
    time_half_life_candidates_weeks: tuple[int | str, ...]
    label_distribution_window_weeks: int
    cohort_scan_batch_size: int
    long_tail_sku_equal_weight: float
    long_tail_sales_band_equal_weight: float

    @property
    def num_folds(self) -> int:
        return self.n_time_blocks - 1


@dataclass(frozen=True)
class DataReaderConfig:
    max_rows_per_read: int
    confirmed_physical_format: str
    partition_availability_granularity: str


@dataclass(frozen=True)
class RestorationConfig:
    abnormal_order_quantile: float
    stockout_min_days_per_week: int
    stockout_flag_share_threshold: float
    shrinkage_k_exposure_days: float
    huber_delta: float
    huber_max_iter: int
    huber_convergence_tolerance: float
    item_effect_ridge: float
    store_effect_ridge: float
    minimum_price_observations: int
    promotion_price_ratio_threshold: float
    stock_grain_stockout_share_threshold: float
    stock_grain_min_open_days: int
    stock_grain_min_positive_sales_days: int


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    spec_version: str
    server_project_root: Path
    data_root: Path
    timezone: str
    log_dir: Path
    probe: ProbeConfig
    data_reader: DataReaderConfig
    training: TrainingConfig
    restoration: RestorationConfig

    @classmethod
    def load(
        cls, project_root: Path, config_path: Path | None = None
    ) -> "AppConfig":
        root = project_root.resolve()
        selected = (config_path or root / "config" / "parameters.yaml").resolve()
        try:
            payload = json.loads(selected.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"Config file not found: {selected}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"parameters.yaml must remain JSON-compatible YAML: {selected}: {exc}"
            ) from exc

        spec_version = str(payload.get("spec_version", ""))
        if spec_version != "3.2":
            raise ConfigError("spec_version must be '3.2'")
        server = payload.get("server", {})
        project = payload.get("project", {})
        probe = payload.get("probe", {})
        data_reader = payload.get("data_reader", {})
        training = payload.get("training", {})
        restoration = payload.get("restoration", {})
        reference_example = training.get("reference_example", {})
        specs_path = Path(probe.get("table_specs_path", "config/source_tables.json"))
        if not specs_path.is_absolute():
            specs_path = root / specs_path
        log_dir = Path(os.environ.get("MINISO_LOG_DIR", project.get("log_dir", "logs")))
        if not log_dir.is_absolute():
            log_dir = root / log_dir
        n_time_blocks = int(training.get("n_time_blocks", 5))
        if n_time_blocks < 2:
            raise ConfigError("training.n_time_blocks must be >= 2")
        label_distribution_window_weeks = int(
            training.get("label_distribution_window_weeks", 26)
        )
        if label_distribution_window_weeks <= 0:
            raise ConfigError(
                "training.label_distribution_window_weeks must be positive"
            )
        cohort_scan_batch_size = int(training.get("cohort_scan_batch_size", 50_000))
        if cohort_scan_batch_size <= 0:
            raise ConfigError("training.cohort_scan_batch_size must be positive")
        train_ratio = float(training.get("train_ratio", 0.8))
        if not 0 < train_ratio < 1:
            raise ConfigError("training.train_ratio must be between 0 and 1")
        sku_equal_share = float(training.get("long_tail_sku_equal_weight", 0.5))
        band_equal_share = float(
            training.get("long_tail_sales_band_equal_weight", 0.5)
        )
        if sku_equal_share < 0 or band_equal_share < 0:
            raise ConfigError("training long-tail weight shares must be non-negative")
        if abs(sku_equal_share + band_equal_share - 1.0) > 1e-12:
            raise ConfigError("training long-tail weight shares must sum to 1")
        sample_files_per_table = int(probe.get("sample_files_per_table", 3))
        max_sample_files_to_attempt = int(
            probe.get("max_sample_files_to_attempt", 50)
        )
        if sample_files_per_table <= 0:
            raise ConfigError("probe.sample_files_per_table must be positive")
        if max_sample_files_to_attempt < sample_files_per_table:
            raise ConfigError(
                "probe.max_sample_files_to_attempt must be >= sample_files_per_table"
            )
        max_rows_per_read = int(data_reader.get("max_rows_per_read", 1_000_000))
        if max_rows_per_read <= 0:
            raise ConfigError("data_reader.max_rows_per_read must be positive")
        confirmed_format = str(
            data_reader.get("confirmed_physical_format", "parquet")
        )
        if confirmed_format != "parquet":
            raise ConfigError(
                "data_reader.confirmed_physical_format must be 'parquet' for ROUND 2"
            )
        availability_granularity = str(
            data_reader.get("partition_availability_granularity", "day")
        )
        if availability_granularity != "day":
            raise ConfigError(
                "data_reader.partition_availability_granularity must be 'day'"
            )
        abnormal_order_quantile = float(
            restoration.get("abnormal_order_quantile", 0.995)
        )
        if not 0 < abnormal_order_quantile < 1:
            raise ConfigError(
                "restoration.abnormal_order_quantile must be between 0 and 1"
            )
        stockout_min_days = int(
            restoration.get("stockout_min_days_per_week", 3)
        )
        if stockout_min_days <= 0:
            raise ConfigError(
                "restoration.stockout_min_days_per_week must be positive"
            )
        stockout_share = float(
            restoration.get("stockout_flag_share_threshold", 0.40)
        )
        if not 0 <= stockout_share <= 1:
            raise ConfigError(
                "restoration.stockout_flag_share_threshold must be between 0 and 1"
            )
        shrinkage_k = float(restoration.get("shrinkage_k_exposure_days", 7.0))
        if shrinkage_k <= 0:
            raise ConfigError(
                "restoration.shrinkage_k_exposure_days must be positive"
            )
        huber_delta = float(restoration.get("huber_delta", 1.345))
        huber_max_iter = int(restoration.get("huber_max_iter", 100))
        huber_convergence_tolerance = float(
            restoration.get("huber_convergence_tolerance", 1e-6)
        )
        if (
            huber_delta <= 0
            or huber_max_iter <= 0
            or huber_convergence_tolerance <= 0
        ):
            raise ConfigError(
                "restoration Huber delta, iteration count, and convergence "
                "tolerance must be positive"
            )
        item_effect_ridge = float(restoration.get("item_effect_ridge", 10.0))
        store_effect_ridge = float(restoration.get("store_effect_ridge", 20.0))
        if item_effect_ridge < 0 or store_effect_ridge < 0:
            raise ConfigError("restoration ridge penalties must be non-negative")
        minimum_price_observations = int(
            restoration.get("minimum_price_observations", 30)
        )
        if minimum_price_observations <= 0:
            raise ConfigError(
                "restoration.minimum_price_observations must be positive"
            )
        promotion_ratio = float(
            restoration.get("promotion_price_ratio_threshold", 0.98)
        )
        stock_grain_share = float(
            restoration.get("stock_grain_stockout_share_threshold", 0.98)
        )
        stock_grain_min_open_days = int(
            restoration.get("stock_grain_min_open_days", 28)
        )
        stock_grain_min_positive_sales_days = int(
            restoration.get("stock_grain_min_positive_sales_days", 3)
        )
        if not 0 <= stock_grain_share <= 1:
            raise ConfigError(
                "restoration.stock_grain_stockout_share_threshold must be "
                "between 0 and 1"
            )
        if (
            stock_grain_min_open_days <= 0
            or stock_grain_min_positive_sales_days <= 0
        ):
            raise ConfigError(
                "restoration stock-grain minimum day counts must be positive"
            )
        if not 0 < promotion_ratio <= 1:
            raise ConfigError(
                "restoration.promotion_price_ratio_threshold must be in (0, 1]"
            )
        return cls(
            project_root=root,
            spec_version=spec_version,
            server_project_root=Path(server.get("project_root", ".")),
            data_root=Path(
                os.environ.get(
                    "MINISO_DATA_ROOT", server.get("data_root", "./data")
                )
            ),
            timezone=os.environ.get(
                "MINISO_TIMEZONE", project.get("timezone", "Asia/Shanghai")
            ),
            log_dir=log_dir,
            probe=ProbeConfig(
                table_specs_path=specs_path,
                sample_files_per_table=sample_files_per_table,
                max_sample_files_to_attempt=max_sample_files_to_attempt,
                sample_rows_per_file=int(probe.get("sample_rows_per_file", 2_000)),
                max_sample_rows_per_table=int(
                    probe.get("max_sample_rows_per_table", 5_000)
                ),
                recent_partition_limit=int(probe.get("recent_partition_limit", 10)),
                sample_file_limit=int(probe.get("sample_file_limit", 10)),
            ),
            data_reader=DataReaderConfig(
                max_rows_per_read=max_rows_per_read,
                confirmed_physical_format=confirmed_format,
                partition_availability_granularity=availability_granularity,
            ),
            training=TrainingConfig(
                train_ratio=train_ratio,
                reference_cohort_size=int(
                    reference_example.get(
                        "cohort_size", training.get("expected_cohort_size", 1178)
                    )
                ),
                reference_train_size=int(
                    reference_example.get(
                        "train_size", training.get("expected_train_size", 955)
                    )
                ),
                reference_test_size=int(
                    reference_example.get(
                        "test_size", training.get("expected_test_size", 223)
                    )
                ),
                reference_train_test_cutoff=date.fromisoformat(
                    str(
                        reference_example.get(
                            "train_test_cutoff",
                            training.get("train_test_cutoff", "2025-11-01"),
                        )
                    )
                ),
                n_time_blocks=n_time_blocks,
                time_half_life_candidates_weeks=tuple(
                    training.get(
                        "time_half_life_candidates_weeks", [26, 52, 78, "infinity"]
                    )
                ),
                label_distribution_window_weeks=label_distribution_window_weeks,
                cohort_scan_batch_size=cohort_scan_batch_size,
                long_tail_sku_equal_weight=sku_equal_share,
                long_tail_sales_band_equal_weight=band_equal_share,
            ),
            restoration=RestorationConfig(
                abnormal_order_quantile=abnormal_order_quantile,
                stockout_min_days_per_week=stockout_min_days,
                stockout_flag_share_threshold=stockout_share,
                shrinkage_k_exposure_days=shrinkage_k,
                huber_delta=huber_delta,
                huber_max_iter=huber_max_iter,
                huber_convergence_tolerance=huber_convergence_tolerance,
                item_effect_ridge=item_effect_ridge,
                store_effect_ridge=store_effect_ridge,
                minimum_price_observations=minimum_price_observations,
                promotion_price_ratio_threshold=promotion_ratio,
                stock_grain_stockout_share_threshold=stock_grain_share,
                stock_grain_min_open_days=stock_grain_min_open_days,
                stock_grain_min_positive_sales_days=(
                    stock_grain_min_positive_sales_days
                ),
            ),
        )


@dataclass(frozen=True)
class TableSpec:
    table_name: str
    partition_column: str
    required_fields: tuple[str, ...]
    read_mode: str = "latest_snapshot"
    business_date_column: str = "ds"
    grain_columns: tuple[str, ...] = field(default_factory=tuple)
    key_columns: tuple[str, ...] = field(default_factory=tuple)
    sku_column: str | None = None
    store_column: str | None = None
    date_columns: tuple[str, ...] = field(default_factory=tuple)
    ip_column: str | None = None


def load_table_specs(path: Path) -> list[TableSpec]:
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Source table spec file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid source table spec JSON: {path}: {exc}") from exc
    rows = payload.get("tables")
    if not isinstance(rows, list) or not rows:
        raise ConfigError("source_tables.json must contain a non-empty 'tables' list")
    specs: list[TableSpec] = []
    seen: set[str] = set()
    for row in rows:
        name = str(row["table_name"])
        if name in seen:
            raise ConfigError(f"Duplicate source table spec: {name}")
        seen.add(name)
        specs.append(
            TableSpec(
                table_name=name,
                partition_column=str(row.get("partition_column", "ds")),
                required_fields=tuple(str(x) for x in row.get("required_fields", [])),
                read_mode=str(row.get("read_mode", "latest_snapshot")),
                business_date_column=str(
                    row.get("business_date_column", row.get("partition_column", "ds"))
                ),
                grain_columns=tuple(str(x) for x in row.get("grain_columns", [])),
                key_columns=tuple(str(x) for x in row.get("key_columns", [])),
                sku_column=row.get("sku_column"),
                store_column=row.get("store_column"),
                date_columns=tuple(str(x) for x in row.get("date_columns", [])),
                ip_column=row.get("ip_column"),
            )
        )
        if specs[-1].read_mode not in {"latest_snapshot", "partition_range"}:
            raise ConfigError(
                f"Unsupported read_mode for {name}: {specs[-1].read_mode}"
            )
    return specs
