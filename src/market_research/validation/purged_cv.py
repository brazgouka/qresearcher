from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np


class PurgedKFold:
    """Time-series-correct k-fold splitting (spec §1.3): folds are contiguous in time
    (never shuffled), any training sample whose label span [t0, t1] overlaps the test
    fold's span is purged, and an additional embargo window immediately after the test
    fold is also dropped from training.

    Both purge and embargo exist because information about the test period can leak
    backward through overlapping label windows and forward through serial correlation
    (Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7).
    """

    def __init__(self, n_splits: int = 5, embargo_frac: float = 0.01) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo_frac = embargo_frac

    def split(self, t0: Sequence[date], t1: Sequence[date]) -> list[tuple[list[int], list[int]]]:
        n = len(t0)
        if len(t1) != n:
            raise ValueError("t0 and t1 must be the same length")

        fold_bounds = np.array_split(np.arange(n), self.n_splits)
        embargo = int(n * self.embargo_frac)

        splits: list[tuple[list[int], list[int]]] = []
        for fold in fold_bounds:
            test_idx = [int(i) for i in fold]
            test_start, test_end = test_idx[0], test_idx[-1]
            test_t0_min = min(t0[i] for i in test_idx)
            test_t1_max = max(t1[i] for i in test_idx)

            train_idx = []
            for i in range(n):
                if test_start <= i <= test_end:
                    continue
                if t0[i] <= test_t1_max and t1[i] >= test_t0_min:
                    continue  # purge: this sample's span overlaps the test fold's span
                if test_end < i <= test_end + embargo:
                    continue  # embargo: immediately follows the test fold
                train_idx.append(i)

            splits.append((train_idx, test_idx))
        return splits
