from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import AppConfig, ConfigError, DatabaseConfig


DB_ENV_NAMES = (
    "MINISO_DB_HOST",
    "MINISO_DB_PORT",
    "MINISO_DB_USER",
    "MINISO_DB_PASSWORD",
    "MINISO_DB_NAME",
)


def test_database_config_requires_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in DB_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigError, match="Missing database environment variables"):
        DatabaseConfig.from_env()


def test_database_config_never_uses_non_miniso_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "MINISO_DB_HOST": "db.internal",
        "MINISO_DB_PORT": "3306",
        "MINISO_DB_USER": "app",
        "MINISO_DB_PASSWORD": "secret",
        "MINISO_DB_NAME": "other",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match="must be 'miniso'"):
        DatabaseConfig.from_env()


def test_database_config_rejects_root(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "MINISO_DB_HOST": "db.internal",
        "MINISO_DB_PORT": "3306",
        "MINISO_DB_USER": "root",
        "MINISO_DB_PASSWORD": "synthetic",
        "MINISO_DB_NAME": "miniso",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match="must not be root"):
        DatabaseConfig.from_env()


def test_v3_2_training_defaults() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = AppConfig.load(project_root)
    assert config.spec_version == "3.2"
    assert str(config.server_project_root) == "/tmp/code/xujing"
    assert config.training.n_time_blocks == 5
    assert config.training.num_folds == 4
    assert config.training.label_distribution_window_weeks == 26
    assert config.training.cohort_scan_batch_size == 50_000
    assert config.training.train_ratio == 0.8
    assert config.training.reference_cohort_size == 1178
    assert config.training.reference_train_size == 955
    assert config.training.reference_test_size == 223
    assert config.training.reference_train_test_cutoff.isoformat() == "2025-11-01"
    assert config.training.long_tail_sku_equal_weight == 0.5
    assert config.training.long_tail_sales_band_equal_weight == 0.5
    assert config.restoration.abnormal_order_quantile == 0.995
    assert config.restoration.stockout_min_days_per_week == 3
    assert config.restoration.stockout_flag_share_threshold == 0.4
    assert config.restoration.shrinkage_k_exposure_days == 7.0
    assert config.restoration.huber_delta == 1.345
    assert config.restoration.minimum_price_observations == 30
