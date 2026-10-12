from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class UnsafeDDL(ValueError):
    """Raised if schema SQL contains anything beyond allowed idempotent creates."""


class SchemaMismatch(RuntimeError):
    """Raised when an existing table differs from the V3.2 contract."""

    def __init__(self, mismatches: Mapping[str, Sequence[str]]) -> None:
        self.mismatches = {name: list(items) for name, items in mismatches.items()}
        detail = "; ".join(
            f"{table}: {', '.join(items)}" for table, items in self.mismatches.items()
        )
        super().__init__(f"SCHEMA_MISMATCH: {detail}")


@dataclass(frozen=True)
class ColumnDefinition:
    name: str
    column_type: str
    nullable: bool
    default: str | None
    auto_increment: bool = False


@dataclass(frozen=True)
class IndexDefinition:
    name: str
    unique: bool
    columns: tuple[str, ...]


@dataclass(frozen=True)
class TableDefinition:
    name: str
    columns: tuple[ColumnDefinition, ...]
    indexes: tuple[IndexDefinition, ...]


ALLOWED_PREFIX = re.compile(r"^CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\b", re.IGNORECASE)
FORBIDDEN_WORDS = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT|REPLACE|RENAME)\b",
    re.IGNORECASE,
)


def load_ddl_statements(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    uncommented = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("--")
    )
    statements = [item.strip() for item in uncommented.split(";") if item.strip()]
    if not statements:
        raise UnsafeDDL(f"No DDL statements found in {path}")
    for statement in statements:
        if not ALLOWED_PREFIX.match(statement):
            raise UnsafeDDL(
                "Only CREATE TABLE IF NOT EXISTS statements are allowed; found: "
                + statement[:80]
            )
        forbidden = FORBIDDEN_WORDS.search(statement)
        if forbidden:
            raise UnsafeDDL(f"Forbidden SQL keyword in schema: {forbidden.group(1)}")
    return statements


def table_name_from_statement(statement: str) -> str:
    match = re.match(
        r"^CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+`?([A-Za-z0-9_]+)`?",
        statement,
        re.IGNORECASE,
    )
    if not match:
        raise UnsafeDDL("Could not parse table name from validated statement")
    return match.group(1)


def _split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for index, character in enumerate(value):
        if quote:
            if character == quote and (index == 0 or value[index - 1] != "\\"):
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _normalize_type(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _normalize_default(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().strip("'").lower()
    return re.sub(r"\s+", "", normalized)


def _parse_index_columns(value: str) -> tuple[str, ...]:
    return tuple(re.sub(r"\s+", "", item.strip(" `")) for item in _split_top_level(value))


def parse_table_definition(statement: str) -> TableDefinition:
    table_name = table_name_from_statement(statement)
    body_start = statement.find("(")
    body_end = statement.rfind(")")
    if body_start < 0 or body_end <= body_start:
        raise UnsafeDDL(f"Could not parse CREATE TABLE body for {table_name}")
    columns: list[ColumnDefinition] = []
    indexes: list[IndexDefinition] = []
    for item in _split_top_level(statement[body_start + 1 : body_end]):
        primary = re.match(r"^PRIMARY\s+KEY\s*\((.*)\)$", item, re.IGNORECASE)
        if primary:
            indexes.append(
                IndexDefinition("PRIMARY", True, _parse_index_columns(primary.group(1)))
            )
            continue
        index_match = re.match(
            r"^(UNIQUE\s+)?KEY\s+`?([A-Za-z0-9_]+)`?\s*\((.*)\)$",
            item,
            re.IGNORECASE,
        )
        if index_match:
            indexes.append(
                IndexDefinition(
                    index_match.group(2),
                    bool(index_match.group(1)),
                    _parse_index_columns(index_match.group(3)),
                )
            )
            continue
        column_match = re.match(r"^`?([A-Za-z0-9_]+)`?\s+(.+)$", item, re.DOTALL)
        if not column_match:
            raise UnsafeDDL(f"Unsupported table item in {table_name}: {item[:100]}")
        name, remainder = column_match.groups()
        constraint = re.search(
            r"\s+(?:NOT\s+NULL|NULL|DEFAULT\b|AUTO_INCREMENT\b|COMMENT\b|COLLATE\b)",
            remainder,
            re.IGNORECASE,
        )
        column_type = remainder[: constraint.start()] if constraint else remainder
        default_match = re.search(
            r"\bDEFAULT\s+('(?:[^']|'')*'|CURRENT_TIMESTAMP(?:\(\d+\))?|[^\s,]+)",
            remainder,
            re.IGNORECASE,
        )
        columns.append(
            ColumnDefinition(
                name=name,
                column_type=_normalize_type(column_type),
                nullable=re.search(r"\bNOT\s+NULL\b", remainder, re.IGNORECASE)
                is None,
                default=_normalize_default(default_match.group(1))
                if default_match
                else None,
                auto_increment=bool(
                    re.search(r"\bAUTO_INCREMENT\b", remainder, re.IGNORECASE)
                ),
            )
        )
    return TableDefinition(table_name, tuple(columns), tuple(indexes))


def parse_schema(statements: Iterable[str]) -> dict[str, TableDefinition]:
    definitions = [parse_table_definition(statement) for statement in statements]
    result = {definition.name: definition for definition in definitions}
    if len(result) != len(definitions):
        raise UnsafeDDL("Duplicate table names in schema")
    return result


def planned_indexes(definitions: Mapping[str, TableDefinition]) -> dict[str, list[str]]:
    return {
        table_name: [index.name for index in definition.indexes]
        for table_name, definition in definitions.items()
    }


def inspect_existing_tables(
    cursor: Any, database: str, table_names: Sequence[str]
) -> dict[str, TableDefinition]:
    if not table_names:
        return {}
    placeholders = ",".join(["%s"] * len(table_names))
    params = [database, *table_names]
    cursor.execute(
        "SELECT TABLE_NAME AS table_name FROM INFORMATION_SCHEMA.TABLES "
        f"WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({placeholders})",
        params,
    )
    existing_names = {str(row["table_name"]) for row in cursor.fetchall()}
    if not existing_names:
        return {}
    existing_placeholders = ",".join(["%s"] * len(existing_names))
    existing_params = [database, *sorted(existing_names)]
    cursor.execute(
        "SELECT TABLE_NAME AS table_name, COLUMN_NAME AS column_name, "
        "COLUMN_TYPE AS column_type, IS_NULLABLE AS is_nullable, "
        "COLUMN_DEFAULT AS column_default, EXTRA AS extra, "
        "ORDINAL_POSITION AS ordinal_position FROM INFORMATION_SCHEMA.COLUMNS "
        f"WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({existing_placeholders}) "
        "ORDER BY TABLE_NAME, ORDINAL_POSITION",
        existing_params,
    )
    columns_by_table: dict[str, list[ColumnDefinition]] = {
        name: [] for name in existing_names
    }
    for row in cursor.fetchall():
        columns_by_table[str(row["table_name"])].append(
            ColumnDefinition(
                name=str(row["column_name"]),
                column_type=_normalize_type(str(row["column_type"])),
                nullable=str(row["is_nullable"]).upper() == "YES",
                default=_normalize_default(row["column_default"]),
                auto_increment="auto_increment" in str(row["extra"]).lower(),
            )
        )
    cursor.execute(
        "SELECT TABLE_NAME AS table_name, INDEX_NAME AS index_name, "
        "NON_UNIQUE AS non_unique, SEQ_IN_INDEX AS seq_in_index, "
        "COLUMN_NAME AS column_name, SUB_PART AS sub_part "
        "FROM INFORMATION_SCHEMA.STATISTICS "
        f"WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({existing_placeholders}) "
        "ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX",
        existing_params,
    )
    index_rows: dict[tuple[str, str], list[tuple[int, str]]] = {}
    index_unique: dict[tuple[str, str], bool] = {}
    for row in cursor.fetchall():
        key = (str(row["table_name"]), str(row["index_name"]))
        column = str(row["column_name"])
        if row["sub_part"] is not None:
            column = f"{column}({int(row['sub_part'])})"
        index_rows.setdefault(key, []).append((int(row["seq_in_index"]), column))
        index_unique[key] = int(row["non_unique"]) == 0
    indexes_by_table: dict[str, list[IndexDefinition]] = {
        name: [] for name in existing_names
    }
    for (table_name, index_name), rows in index_rows.items():
        indexes_by_table[table_name].append(
            IndexDefinition(
                name=index_name,
                unique=index_unique[(table_name, index_name)],
                columns=tuple(column for _, column in sorted(rows)),
            )
        )
    return {
        name: TableDefinition(
            name=name,
            columns=tuple(columns_by_table[name]),
            indexes=tuple(indexes_by_table[name]),
        )
        for name in existing_names
    }


def compare_table_schema(
    expected: TableDefinition, actual: TableDefinition
) -> list[str]:
    problems: list[str] = []
    expected_columns = {column.name: column for column in expected.columns}
    actual_columns = {column.name: column for column in actual.columns}
    missing_columns = sorted(set(expected_columns) - set(actual_columns))
    extra_columns = sorted(set(actual_columns) - set(expected_columns))
    if missing_columns:
        problems.append("missing columns=" + ",".join(missing_columns))
    if extra_columns:
        problems.append("unexpected columns=" + ",".join(extra_columns))
    for name in sorted(set(expected_columns) & set(actual_columns)):
        planned = expected_columns[name]
        observed = actual_columns[name]
        if planned.column_type != observed.column_type:
            problems.append(
                f"column {name} type expected={planned.column_type} actual={observed.column_type}"
            )
        if planned.nullable != observed.nullable:
            problems.append(
                f"column {name} nullable expected={planned.nullable} actual={observed.nullable}"
            )
        if planned.default != observed.default:
            problems.append(
                f"column {name} default expected={planned.default} actual={observed.default}"
            )
        if planned.auto_increment != observed.auto_increment:
            problems.append(
                f"column {name} auto_increment expected={planned.auto_increment} "
                f"actual={observed.auto_increment}"
            )
    expected_indexes = {index.name: index for index in expected.indexes}
    actual_indexes = {index.name: index for index in actual.indexes}
    for name in sorted(set(expected_indexes) - set(actual_indexes)):
        problems.append(f"missing index={name}")
    for name in sorted(set(expected_indexes) & set(actual_indexes)):
        planned = expected_indexes[name]
        observed = actual_indexes[name]
        if planned.unique != observed.unique or planned.columns != observed.columns:
            problems.append(
                f"index {name} expected unique={planned.unique} columns={planned.columns} "
                f"actual unique={observed.unique} columns={observed.columns}"
            )
    return problems


def find_schema_mismatches(
    expected: Mapping[str, TableDefinition],
    actual: Mapping[str, TableDefinition],
    *,
    require_all_tables: bool,
) -> dict[str, list[str]]:
    mismatches: dict[str, list[str]] = {}
    for table_name, planned in expected.items():
        observed = actual.get(table_name)
        if observed is None:
            if require_all_tables:
                mismatches[table_name] = ["table missing"]
            continue
        problems = compare_table_schema(planned, observed)
        if problems:
            mismatches[table_name] = problems
    return mismatches
