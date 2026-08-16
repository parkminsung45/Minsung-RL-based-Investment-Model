import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.trading_env import PortfolioEnv


def _make_env(
    n_days=40, n_assets=2, window=5, episode_length=10,
    transaction_cost_bps=0.0, random_start=False, n_features=3, seed=0,
):
    tickers = [f"T{i}" for i in range(n_assets)]
    dates = pd.bdate_range("2024-01-02", periods=n_days)

    rng = np.random.default_rng(seed)
    close_df = pd.DataFrame(
        {t: 100 + np.cumsum(rng.normal(0, 1, n_days)) for t in tickers}, index=dates
    )
    feature_cols = pd.MultiIndex.from_product([tickers, [f"f{i}" for i in range(n_features)]])
    feature_df = pd.DataFrame(
        rng.normal(0, 1, (n_days, n_assets * n_features)), index=dates, columns=feature_cols
    )

    env = PortfolioEnv(
        feature_df, close_df, tickers, window=window, episode_length=episode_length,
        transaction_cost_bps=transaction_cost_bps, random_start=random_start, seed=seed,
    )
    return env, tickers, close_df


def test_reset_starts_all_cash():
    env, _, _ = _make_env()
    env.reset()

    assert env._weights[-1] == pytest.approx(1.0)
    assert env._weights[:-1].sum() == pytest.approx(0.0)


def test_action_normalizes_to_weights_summing_to_one():
    env, tickers, _ = _make_env()
    env.reset()

    action = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    _, _, _, _, info = env.step(action)

    assert info["weights"].sum() == pytest.approx(1.0, abs=1e-5)
    assert (info["weights"] >= 0).all()


def test_portfolio_value_matches_manual_calculation_without_cost():
    env, tickers, close_df = _make_env(transaction_cost_bps=0.0)
    env.reset()
    start_idx = env.window

    action = np.array([2.0, 0.0, -2.0], dtype=np.float32)
    weights = PortfolioEnv._softmax(action)
    _, reward, _, _, info = env.step(action)

    prices_now = close_df.iloc[start_idx][tickers].values
    prices_next = close_df.iloc[start_idx + 1][tickers].values
    asset_returns = (prices_next / prices_now) - 1.0
    expected_return = float(np.dot(weights[:-1], asset_returns))

    assert info["portfolio_value"] == pytest.approx(1.0 * (1 + expected_return), rel=1e-5)
    assert reward == pytest.approx(np.log(1 + expected_return), rel=1e-5)


def test_transaction_cost_reduces_value_versus_no_cost():
    env_no_cost, tickers, _ = _make_env(transaction_cost_bps=0.0)
    env_with_cost, _, _ = _make_env(transaction_cost_bps=100.0)  # 1%
    env_no_cost.reset()
    env_with_cost.reset()

    action = np.array([2.0, 0.0, -2.0], dtype=np.float32)
    _, _, _, _, info_no_cost = env_no_cost.step(action)
    _, _, _, _, info_with_cost = env_with_cost.step(action)

    assert info_with_cost["portfolio_value"] < info_no_cost["portfolio_value"]


def test_episode_terminates_after_episode_length_steps():
    env, tickers, _ = _make_env(n_days=40, window=5, episode_length=10, random_start=False)
    env.reset()

    action = np.zeros(len(tickers) + 1, dtype=np.float32)
    terminated = False
    steps = 0
    while not terminated:
        _, _, terminated, _, _ = env.step(action)
        steps += 1
        assert steps <= 10  # 무한루프 방지 안전장치

    assert steps == 10


def test_rejects_mismatched_indexes():
    tickers = ["A", "B"]
    dates_a = pd.bdate_range("2024-01-02", periods=30)
    dates_b = pd.bdate_range("2024-02-02", periods=30)
    feature_cols = pd.MultiIndex.from_product([tickers, ["f0"]])
    feature_df = pd.DataFrame(0.0, index=dates_a, columns=feature_cols)
    close_df = pd.DataFrame(100.0, index=dates_b, columns=tickers)

    with pytest.raises(ValueError):
        PortfolioEnv(feature_df, close_df, tickers, window=5, episode_length=5)
