from __future__ import annotations

from pathlib import Path

import pytest

from db.schema import (
    ColumnDefinition,
    IndexDefinition,
    TableDefinition,
    UnsafeDDL,
    compare_table_schema,
    load_ddl_statements,
    parse_schema,
    table_name_from_statement,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v3_2_schema_contains_only_the_nine_expected_tables() -> None:
    statements = load_ddl_statements(PROJECT_ROOT / "sql" / "schema_v3_2.sql")
    names = {table_name_from_statement(statement) for statement in statements}
    assert names == {
        "algo_decision_run",
        "algo_feature_snapshot",
        "algo_candidate_detail",
        "algo_calc_detail",
        "algo_prediction",
        "human_decision",
        "supply_execution",
        "sku_store_daily",
        "evaluation_label",
    }
    definitions = parse_schema(statements)
    run_columns = {column.name for column in definitions["algo_decision_run"].columns}
    assert "superseded_run_id" in run_columns
    assert "supersedes_run_id" not in run_columns


def test_ddl_guard_rejects_destructive_sql(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.sql"
    path.write_text("DROP TABLE algo_prediction;", encoding="utf-8")
    with pytest.raises(UnsafeDDL):
        load_ddl_statements(path)


def test_schema_comparison_detects_type_and_index_mismatch() -> None:
    expected = TableDefinition(
        name="example",
        columns=(ColumnDefinition("id", "bigint unsigned", False, None, True),),
        indexes=(IndexDefinition("PRIMARY", True, ("id",)),),
    )
    actual = TableDefinition(
        name="example",
        columns=(ColumnDefinition("id", "bigint", False, None, False),),
        indexes=(),
    )
    problems = compare_table_schema(expected, actual)
    assert any("type expected" in problem for problem in problems)
    assert any("auto_increment" in problem for problem in problems)
    assert "missing index=PRIMARY" in problems
