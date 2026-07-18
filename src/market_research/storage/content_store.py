from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl


def content_hash(source: str, params: dict[str, Any]) -> str:
    """Deterministic hash of (source, params) — identical inputs always hash identically.

    Reproducibility (spec §1.7) depends on this: a result must be re-derivable from its
    logged config, so the address a dataset lives at must be a pure function of what
    produced it, not of when it was written.
    """
    canonical = json.dumps({"source": source, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class ContentStore:
    """Content-addressed Parquet store.

    Writing the same (source, params) twice resolves to the same path — callers get
    free memoization and an audit trail of exactly what config produced each file.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, source: str, params: dict[str, Any]) -> Path:
        return self.root / source / f"{content_hash(source, params)}.parquet"

    def exists(self, source: str, params: dict[str, Any]) -> bool:
        return self.path_for(source, params).exists()

    def write(self, source: str, params: dict[str, Any], frame: pl.DataFrame) -> Path:
        path = self.path_for(source, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        return path

    def read(self, source: str, params: dict[str, Any]) -> pl.DataFrame:
        path = self.path_for(source, params)
        if not path.exists():
            raise FileNotFoundError(
                f"No stored data for source={source!r}, params={params!r} at {path}"
            )
        return pl.read_parquet(path)
