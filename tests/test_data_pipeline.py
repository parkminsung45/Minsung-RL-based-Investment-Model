import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data_pipeline as dp


@pytest.fixture(autouse=True)
def isolated_universe_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "_CACHE_PATH", str(tmp_path / "universe_cache.json"))
    monkeypatch.setattr(dp, "_CACHE_DIR", str(tmp_path))


def test_get_tickers_uses_first_successful_fetcher():
    result = dp._get_universe_tickers("test", [lambda: ["AAPL", "MSFT"]])
    assert result == ["AAPL", "MSFT"]


def test_get_tickers_falls_back_to_second_fetcher_on_failure():
    def failing():
        raise RuntimeError("boom")

    result = dp._get_universe_tickers("test", [failing, lambda: ["NVDA"]])
    assert result == ["NVDA"]


def test_get_tickers_caches_successful_result():
    dp._get_universe_tickers("test", [lambda: ["AAPL"]])
    cache = dp._load_universe_cache()
    assert cache["test"]["tickers"] == ["AAPL"]
    assert "fetched_at" in cache["test"]


def test_get_tickers_falls_back_to_cache_when_all_fetchers_fail():
    dp._get_universe_tickers("test", [lambda: ["AAPL", "MSFT"]])

    def failing():
        raise RuntimeError("network down")

    result = dp._get_universe_tickers("test", [failing])
    assert result == ["AAPL", "MSFT"]


def test_get_tickers_raises_when_no_fetcher_and_no_cache():
    def failing():
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError):
        dp._get_universe_tickers("nonexistent", [failing])


def test_get_universe_tickers_unions_and_dedupes(monkeypatch):
    monkeypatch.setattr(dp, "get_sp500_tickers", lambda: ["AAPL", "MSFT", "NVDA"])
    monkeypatch.setattr(dp, "get_nasdaq100_tickers", lambda: ["MSFT", "GOOGL"])

    result = dp.get_universe_tickers()
    assert result == ["AAPL", "GOOGL", "MSFT", "NVDA"]
