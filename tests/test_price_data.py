import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.price_data import FEATURE_COLUMNS, compute_features


def _synthetic_price_df(tickers, n=100, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    frames = {}
    for i, ticker in enumerate(tickers):
        close = 100 + np.cumsum(rng.normal(0, 1, n)) + i * 10
        volume = rng.integers(1_000_000, 2_000_000, n).astype(float)
        frames[ticker] = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": volume},
            index=dates,
        )
    return pd.concat(frames, axis=1)


def test_compute_features_columns_and_no_nan():
    tickers = ["AAA", "BBB"]
    price_df = _synthetic_price_df(tickers, n=100)

    features = compute_features(price_df, tickers)

    assert not features.isna().any().any()
    assert set(features.columns.get_level_values(0)) == set(tickers)
    for ticker in tickers:
        assert list(features[ticker].columns) == FEATURE_COLUMNS


def test_compute_features_drops_warmup_rows():
    tickers = ["AAA"]
    price_df = _synthetic_price_df(tickers, n=100)

    features = compute_features(price_df, tickers)

    # ma60_return 등 60일 롤링 지표 때문에 최소 55행 이상은 드롭되어야 한다.
    assert 0 < len(features) < len(price_df) - 55


def test_return_1d_matches_manual_calculation():
    tickers = ["AAA"]
    price_df = _synthetic_price_df(tickers, n=100)
    expected_return = price_df[("AAA", "Close")].pct_change()

    features = compute_features(price_df, tickers)

    pd.testing.assert_series_equal(
        features[("AAA", "return_1d")],
        expected_return.loc[features.index],
        check_names=False,
    )


def test_rsi_within_bounds():
    tickers = ["AAA"]
    price_df = _synthetic_price_df(tickers, n=100)

    features = compute_features(price_df, tickers)
    rsi = features[("AAA", "rsi_14")]

    assert (rsi >= 0).all() and (rsi <= 100).all()
