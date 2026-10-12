from __future__ import annotations

import pytest

from config.settings import DatabaseConfig
from db.connection import Database


class FakeCursor:
    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeConnection:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0
        self.closed = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor()

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed += 1


class FakeDatabase(Database):
    def __init__(self, connection: FakeConnection) -> None:
        super().__init__(DatabaseConfig("host", 3306, "user", "password", "miniso"))
        self.connection = connection

    def connect(self, *, autocommit: bool = False) -> FakeConnection:
        assert autocommit is False
        return self.connection


def test_transaction_commits_on_success() -> None:
    connection = FakeConnection()
    with FakeDatabase(connection).transaction() as cursor:
        assert isinstance(cursor, FakeCursor)
    assert (connection.commits, connection.rollbacks, connection.closed) == (1, 0, 1)


def test_transaction_rolls_back_on_error() -> None:
    connection = FakeConnection()
    with pytest.raises(RuntimeError):
        with FakeDatabase(connection).transaction():
            raise RuntimeError("synthetic failure")
    assert (connection.commits, connection.rollbacks, connection.closed) == (0, 1, 1)
