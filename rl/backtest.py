"""
학습된 PPO 포트폴리오 에이전트를 보류(holdout) 구간에 결정적으로 실행해
동일종목 동일가중 buy&hold, 전량 현금 보유와 성과를 비교한다.

실행 방법:
    python -m rl.backtest
"""
import json
import os
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import config
from rl.price_data import load_dataset
from rl.trading_env import PortfolioEnv


def _meta_path(model_path: str) -> str:
    return os.path.splitext(model_path)[0] + ".meta.json"


def load_holdout(meta: Dict, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    학습 시 보류해둔 구간(split_date 이후)을 다시 불러온다. 오늘 날짜까지 새로
    조회하므로, train.py 실행 이후 시간이 지났다면 보류 구간이 더 길어져 있다.
    """
    feature_df, close_df = load_dataset(meta["tickers"], start=meta["train_start"], use_cache=use_cache)
    split_idx = feature_df.index.searchsorted(pd.Timestamp(meta["split_date"]))
    holdout_start = max(split_idx - meta["window"], 0)
    return feature_df.iloc[holdout_start:], close_df.iloc[holdout_start:]


def compute_metrics(values: np.ndarray, periods_per_year: int = 252) -> Dict[str, float]:
    returns = np.diff(values) / values[:-1]
    n_periods = max(len(values) - 1, 1)
    cagr = (values[-1] / values[0]) ** (periods_per_year / n_periods) - 1
    ann_vol = float(returns.std() * np.sqrt(periods_per_year))
    sharpe = float((returns.mean() * periods_per_year) / ann_vol) if ann_vol > 0 else 0.0
    running_max = np.maximum.accumulate(values)
    max_drawdown = float(((values - running_max) / running_max).min())
    return {
        "total_return": float(values[-1] / values[0] - 1),
        "cagr": float(cagr),
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def _run_policy(model, feature_df, close_df, tickers, window, transaction_cost_bps):
    episode_length = len(feature_df) - window - 1
    env = PortfolioEnv(
        feature_df, close_df, tickers, window=window, episode_length=episode_length,
        transaction_cost_bps=transaction_cost_bps, random_start=False,
    )
    obs, _ = env.reset()
    values = [1.0]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        values.append(info["portfolio_value"])
        done = terminated or truncated

    values = np.array(values)
    dates = feature_df.index[window: window + len(values)]
    return values, dates


def _equal_weight_baseline(close_df, tickers, window, n_steps):
    prices = close_df[tickers].iloc[window: window + n_steps]
    normalized = prices / prices.iloc[0]
    return normalized.mean(axis=1).values


def run_backtest(model, feature_df, close_df, tickers, window, transaction_cost_bps=10.0) -> Dict:
    """
    RL 정책과 두 베이스라인(동일가중 buy&hold, 현금)의 자산곡선·성과지표를 반환한다.
    daily_run.py의 드리프트 점검에서도 이 함수를 재사용한다.
    """
    rl_values, dates = _run_policy(model, feature_df, close_df, tickers, window, transaction_cost_bps)
    bh_values = _equal_weight_baseline(close_df, tickers, window, len(rl_values))
    cash_values = np.ones(len(rl_values))

    return {
        "dates": list(dates),
        "rl": {"values": rl_values, "metrics": compute_metrics(rl_values)},
        "buy_and_hold": {"values": bh_values, "metrics": compute_metrics(bh_values)},
        "cash": {"values": cash_values, "metrics": compute_metrics(cash_values)},
    }


def _plot_equity_curves(result: Dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # matplotlib 기본 폰트(DejaVu Sans)가 한글 글리프를 지원하지 않아 라벨은 영문으로 표기.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result["dates"], result["rl"]["values"], label="RL Portfolio")
    ax.plot(result["dates"], result["buy_and_hold"]["values"], label="Equal-Weight Buy&Hold")
    ax.plot(result["dates"], result["cash"]["values"], label="Cash", linestyle="--")
    ax.set_ylabel("Portfolio Value (normalized, start=1.0)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(model_path: str = None) -> Dict:
    model_path = model_path or config.RL_MODEL_PATH
    with open(_meta_path(model_path), encoding="utf-8") as f:
        meta = json.load(f)

    print(f"[1/3] 모델 로드 중... {model_path}")
    model = PPO.load(model_path)

    print(f"[2/3] 보류 구간 백테스트 실행 중... (split_date={meta['split_date']} 이후)")
    feature_df, close_df = load_holdout(meta)
    result = run_backtest(model, feature_df, close_df, meta["tickers"], meta["window"], meta["transaction_cost_bps"])

    for name, label in [("rl", "RL 포트폴리오"), ("buy_and_hold", "동일가중 Buy&Hold"), ("cash", "현금")]:
        m = result[name]["metrics"]
        print(
            f"  [{label}] CAGR={m['cagr']:.2%} 연변동성={m['annual_volatility']:.2%} "
            f"Sharpe={m['sharpe']:.2f} MDD={m['max_drawdown']:.2%}"
        )

    print("[3/3] 결과 저장 중...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    today = date.today().isoformat()
    csv_path = os.path.join(config.OUTPUT_DIR, f"rl_backtest_{today}.csv")
    pd.DataFrame({
        "date": result["dates"],
        "rl_value": result["rl"]["values"],
        "buy_and_hold_value": result["buy_and_hold"]["values"],
        "cash_value": result["cash"]["values"],
    }).to_csv(csv_path, index=False)

    png_path = os.path.join(config.OUTPUT_DIR, f"rl_backtest_{today}.png")
    _plot_equity_curves(result, png_path)

    print(f"완료: {csv_path}, {png_path}")
    return result


if __name__ == "__main__":
    main()
