#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.logging_utils import log_exception, setup_run_logger  # noqa: E402
from config.settings import AppConfig, ConfigError, DatabaseConfig  # noqa: E402
from db.connection import Database  # noqa: E402
from db.schema import (  # noqa: E402
    SchemaMismatch,
    find_schema_mismatches,
    inspect_existing_tables,
    load_ddl_statements,
    parse_schema,
    planned_indexes,
    table_name_from_statement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and validate the nine V3.2 governance/result tables."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-skus", type=int, default=None)
    parser.add_argument("--sku-code", action="append", default=[])
    parser.add_argument(
        "--schema", type=Path, default=PROJECT_ROOT / "sql" / "schema_v3_2.sql"
    )
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(PROJECT_ROOT, args.config)
    logger, log_path = setup_run_logger("init_database", config.log_dir, config.timezone)
    try:
        statements = load_ddl_statements(args.schema)
        table_names = [table_name_from_statement(statement) for statement in statements]
        definitions = parse_schema(statements)
        if args.dry_run:
            database_name = os.environ.get("MINISO_DB_NAME", "miniso")
            if database_name != "miniso":
                raise ConfigError("MINISO_DB_NAME must be 'miniso' for this project")
            print("DRY RUN: no database connection will be opened.")
            print(f"database: {database_name}")
            print(f"planned tables ({len(table_names)}):")
            indexes = planned_indexes(definitions)
            for table_name in table_names:
                print(f"  - {table_name}")
                print(f"    indexes: {', '.join(indexes[table_name])}")
            print(f"\nlog: {log_path}")
            logger.info(
                "dry-run database=%s validated tables=%s",
                database_name,
                ",".join(table_names),
            )
            return 0

        database_config = DatabaseConfig.from_env()
        database = Database(database_config)
        logger.info(
            "initializing database=%s host=%s tables=%s",
            database_config.database,
            database_config.host,
            ",".join(table_names),
        )
        with database.transaction() as cursor:
            observed_before = inspect_existing_tables(
                cursor, database_config.database, table_names
            )
            mismatches = find_schema_mismatches(
                definitions, observed_before, require_all_tables=False
            )
            if mismatches:
                raise SchemaMismatch(mismatches)
            for table_name, statement in zip(table_names, statements):
                if table_name in observed_before:
                    logger.info("existing schema validated table=%s", table_name)
                    continue
                logger.info("executing CREATE TABLE IF NOT EXISTS table=%s", table_name)
                cursor.execute(statement)
            observed_after = inspect_existing_tables(
                cursor, database_config.database, table_names
            )
            mismatches = find_schema_mismatches(
                definitions, observed_after, require_all_tables=True
            )
            if mismatches:
                raise SchemaMismatch(mismatches)
        print(
            f"SCHEMA_OK: validated {len(statements)} V3.2 tables in "
            f"database {database_config.database}."
        )
        print(f"log: {log_path}")
        return 0
    except SchemaMismatch as exc:
        log_exception(logger, module="init_database.schema_validation", exc=exc)
        print(str(exc), file=sys.stderr)
        print("No existing table was dropped or altered.", file=sys.stderr)
        print(f"log: {log_path}", file=sys.stderr)
        return 2
    except Exception as exc:
        log_exception(logger, module="init_database", exc=exc)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"log: {log_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
