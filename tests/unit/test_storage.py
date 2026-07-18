from __future__ import annotations

import polars as pl
import pytest

from market_research.storage import ContentStore, DuckDBStore, content_hash


@pytest.fixture
def sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "close": [100.0, 101.5, 99.25],
        }
    )


def test_content_hash_is_deterministic() -> None:
    params = {"symbol": "AAPL", "start": "2024-01-01", "end": "2024-01-03"}
    assert content_hash("ohlcv", params) == content_hash("ohlcv", dict(params))


def test_content_hash_differs_on_params_or_source() -> None:
    base = content_hash("ohlcv", {"symbol": "AAPL"})
    assert base != content_hash("ohlcv", {"symbol": "MSFT"})
    assert base != content_hash("other_source", {"symbol": "AAPL"})


def test_content_hash_ignores_key_order() -> None:
    a = content_hash("ohlcv", {"symbol": "AAPL", "start": "2024-01-01"})
    b = content_hash("ohlcv", {"start": "2024-01-01", "symbol": "AAPL"})
    assert a == b


def test_write_then_read_round_trips(tmp_path, sample_frame: pl.DataFrame) -> None:
    store = ContentStore(tmp_path)
    params = {"symbol": "AAPL"}

    assert not store.exists("ohlcv", params)
    store.write("ohlcv", params, sample_frame)
    assert store.exists("ohlcv", params)

    result = store.read("ohlcv", params)
    assert result.equals(sample_frame)


def test_same_params_resolve_to_same_path(tmp_path, sample_frame: pl.DataFrame) -> None:
    store = ContentStore(tmp_path)
    params = {"symbol": "AAPL"}

    path_a = store.write("ohlcv", params, sample_frame)
    path_b = store.write("ohlcv", dict(params), sample_frame)
    assert path_a == path_b


def test_different_params_resolve_to_different_paths(tmp_path, sample_frame: pl.DataFrame) -> None:
    store = ContentStore(tmp_path)
    path_a = store.write("ohlcv", {"symbol": "AAPL"}, sample_frame)
    path_b = store.write("ohlcv", {"symbol": "MSFT"}, sample_frame)
    assert path_a != path_b


def test_read_missing_raises(tmp_path) -> None:
    store = ContentStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.read("ohlcv", {"symbol": "NOPE"})


def test_duckdb_store_queries_written_parquet(tmp_path, sample_frame: pl.DataFrame) -> None:
    content_store = ContentStore(tmp_path)
    content_store.write("ohlcv", {"symbol": "AAPL"}, sample_frame)

    db = DuckDBStore()
    db.register_parquet_glob("ohlcv_view", str(tmp_path / "ohlcv" / "*.parquet"))
    result = db.query("SELECT * FROM ohlcv_view ORDER BY date")

    assert result.sort("date").equals(sample_frame.sort("date"))
    db.close()


def test_duckdb_store_rejects_invalid_view_name(tmp_path) -> None:
    db = DuckDBStore()
    with pytest.raises(ValueError):
        db.register_parquet_glob("bad; DROP TABLE x", str(tmp_path / "*.parquet"))
    db.close()
