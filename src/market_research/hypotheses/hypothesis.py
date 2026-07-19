from __future__ import annotations

from typing import Protocol

import polars as pl


class Hypothesis(Protocol):
    """A composable predicate over a features frame that emits positions.

    Every node in the tree — leaf predicate or AND/OR/NOT combinator — implements this
    same interface, so hypotheses are built by composition rather than hardcoded as
    strategy classes (spec §4). `positions` returns a 0.0/1.0-valued series (flat/long)
    per row; combinators combine child position series with boolean algebra.

    `config()` must return a JSON-serializable, deterministic description of the whole
    tree — it's what the trial registry hashes for content-addressing (spec §1.7), so a
    hypothesis's identity is exactly its config, not its Python object identity.
    """

    def positions(self, features: pl.DataFrame) -> pl.Series: ...

    def complexity(self) -> int: ...

    def config(self) -> dict: ...
