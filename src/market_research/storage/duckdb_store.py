from __future__ import annotations

import re
from pathlib import Path

import duckdb
import polars as pl

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DuckDBStore:
    """SQL access over content-addressed Parquet via an embedded DuckDB connection.

    No server process — matches the "small correct system" preference (spec §9.5) and
    keeps registry/analytics queries on the same storage the ContentStore already writes.
    """

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.conn = duckdb.connect(str(db_path))

    def register_parquet_glob(self, view_name: str, glob_pattern: str) -> None:
        if not _IDENTIFIER_RE.match(view_name):
            raise ValueError(f"Invalid view name: {view_name!r}")
        # DuckDB can't prepare-parameterize DDL statements, so the glob is escaped and inlined.
        escaped_glob = glob_pattern.replace("'", "''")
        self.conn.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{escaped_glob}')"
        )

    def query(self, sql: str, params: list | None = None) -> pl.DataFrame:
        return self.conn.execute(sql, params or []).pl()

    def execute(self, sql: str, params: list | None = None) -> None:
        self.conn.execute(sql, params or [])

    def close(self) -> None:
        self.conn.close()
