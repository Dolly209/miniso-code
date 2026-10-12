from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from config.settings import DatabaseConfig


class DatabaseDependencyError(RuntimeError):
    """Raised when the server DB driver is unavailable."""


def _pymysql() -> Any:
    try:
        import pymysql
    except ImportError as exc:
        raise DatabaseDependencyError(
            "PyMySQL is required for database access; install server requirements first"
        ) from exc
    return pymysql


class Database:
    """Small connection/transaction boundary; credentials always come from env config."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self.config = config or DatabaseConfig.from_env()

    def connect(self, *, autocommit: bool = False) -> Any:
        pymysql = _pymysql()
        return pymysql.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            charset="utf8mb4",
            autocommit=autocommit,
            connect_timeout=10,
            read_timeout=30,
            write_timeout=30,
            cursorclass=pymysql.cursors.DictCursor,
        )

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        connection = self.connect(autocommit=False)
        try:
            with connection.cursor() as cursor:
                yield cursor
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def ping(self) -> dict[str, Any]:
        connection = self.connect(autocommit=True)
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT VERSION() AS version")
                version = cursor.fetchone()["version"]
                cursor.execute("SELECT DATABASE() AS database_name")
                database_name = cursor.fetchone()["database_name"]
                cursor.execute("SELECT 1 AS ok")
                ok = cursor.fetchone()["ok"]
            return {"version": version, "database": database_name, "select_1": ok}
        finally:
            connection.close()
