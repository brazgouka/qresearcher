from __future__ import annotations

from datetime import date, timedelta

import pytest

from market_research.validation import PurgedKFold


def _dates(n: int) -> list[date]:
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]


def test_test_folds_partition_the_full_index_range() -> None:
    n = 20
    t0 = _dates(n)
    t1 = _dates(n)  # zero-length spans: t1 == t0
    splits = PurgedKFold(n_splits=4, embargo_frac=0.0).split(t0, t1)

    all_test_idx = sorted(i for _, test in splits for i in test)
    assert all_test_idx == list(range(n))


def test_train_excludes_its_own_test_fold() -> None:
    n = 20
    t0 = _dates(n)
    t1 = _dates(n)
    splits = PurgedKFold(n_splits=4, embargo_frac=0.0).split(t0, t1)

    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(test_idx)


def test_purges_training_samples_whose_span_overlaps_test_fold() -> None:
    # every label spans the entire series -> every training sample overlaps every test fold
    n = 10
    t0 = _dates(n)
    t1 = [t0[-1]] * n  # every span ends at the last date -> overlaps everything before it
    splits = PurgedKFold(n_splits=2, embargo_frac=0.0).split(t0, t1)

    first_train, first_test = splits[0]
    # samples after the first test fold still overlap it (their t1 all equal the last date)
    # and samples before/within it trivially overlap -> train should be empty or near-empty
    test_t1_max = max(t1[j] for j in first_test)
    test_t0_min = min(t0[j] for j in first_test)
    for i in first_train:
        overlaps = t0[i] <= test_t1_max and t1[i] >= test_t0_min
        assert not overlaps


def test_embargo_drops_samples_immediately_after_test_fold() -> None:
    n = 20
    t0 = _dates(n)
    t1 = _dates(n)
    splits = PurgedKFold(n_splits=4, embargo_frac=0.25).split(t0, t1)  # embargo = 5 samples

    train_idx, test_idx = splits[0]
    test_end = test_idx[-1]
    embargoed = set(range(test_end + 1, test_end + 1 + 5))
    assert embargoed.isdisjoint(train_idx)


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=2).split(_dates(5), _dates(4))


def test_rejects_too_few_splits() -> None:
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)
