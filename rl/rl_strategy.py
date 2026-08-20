"""
학습된 PPO 포트폴리오 에이전트로 오늘자 목표 비중을 계산해, 보유 종목을
목표 비중에 맞춰 리밸런싱하는 실거래 연동 모듈. 이 저장소의 유일한 매매
전략 경로다(과거 규칙 기반 strategy.py/뉴스+애널리스트 감성분석 파이프라인은
매크로 블로그 기반 RL로 대체되어 제거됨).

broker.create_order()의 기존 이중 안전장치(config.TOSS_LIVE_TRADING=false
기본 드라이런, confirm=True 명시 필요)를 그대로 사용한다. 이 모듈은 실거래
여부를 직접 판단하지 않는다.
"""
import json
import os
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import broker
import config
from rl import daily_log, macro_blog, paper_trading
from rl.price_data import load_dataset
from rl.trading_env import PortfolioEnv

# 실거래 관측 시 최근 몇 달치 데이터를 불러올지 (관측 윈도 + 60일 이동평균
# 워밍업을 넉넉히 덮도록 여유 있게 잡음)
_INFERENCE_LOOKBACK_DAYS = 250


def _meta_path(model_path: str) -> str:
    return os.path.splitext(model_path)[0] + ".meta.json"


def load_model_and_meta(model_path: Optional[str] = None) -> Tuple[PPO, Dict]:
    model_path = model_path or config.RL_MODEL_PATH
    with open(_meta_path(model_path), encoding="utf-8") as f:
        meta = json.load(f)
    return PPO.load(model_path), meta


def _build_observation(
    feature_df, tickers: List[str], window: int, current_weights: np.ndarray,
    macro_window: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_assets = len(tickers)
    n_features = feature_df.shape[1] // n_assets
    window_slice = feature_df.tail(window).values.astype(np.float32).reshape(window, n_assets, n_features)
    parts = [window_slice.reshape(-1)]
    if macro_window is not None:
        parts.append(macro_window.astype(np.float32))
    parts.append(current_weights)
    return np.concatenate(parts).astype(np.float32)


def _macro_window(feature_df, window: int) -> np.ndarray:
    """feature_df의 마지막 window일에 맞춰 매크로 점수 윈도를 만든다. 학습 때와
    똑같이 결측일은 직전값으로 이어붙이고(ffill), 그마저 없으면 중립값 0.0.
    daily_run은 30분마다 도는데, 여기서는 저장소(rl/macro_daily_scores.json)만
    읽을 뿐 크롤링은 하지 않는다(crawling/scoring은 10시 launchd 잡 전담)."""
    dates = feature_df.index[-window:]
    scores = macro_blog.load_daily_scores()
    series = pd.Series(scores, dtype=float)
    series.index = pd.to_datetime(series.index)
    aligned = series.reindex(dates).ffill().fillna(0.0)
    return aligned.values


def _holdings_by_symbol(client: broker.TossClient) -> Dict[str, float]:
    """실계좌 검증 완료(2026-08-17): 보유종목은 result.items, quantity는 문자열."""
    qty_by_symbol = {}
    for item in broker.holdings_items(client):
        symbol = item.get("symbol")
        qty = item.get("quantity")
        if symbol and qty is not None and float(qty) > 0:
            qty_by_symbol[symbol] = float(qty)
    return qty_by_symbol


def _current_weights(qty_by_symbol: Dict[str, float], latest_close: Dict[str, float], cash: float, tickers: List[str]) -> np.ndarray:
    asset_values = [qty_by_symbol.get(t, 0.0) * latest_close[t] for t in tickers]
    total = sum(asset_values) + cash
    if total <= 0:
        weights = [0.0] * len(tickers) + [1.0]
    else:
        weights = [v / total for v in asset_values] + [cash / total]
    return np.array(weights, dtype=np.float32)


def decide_target_weights(model: PPO, meta: Dict, current_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    가장 최근 시장 데이터로 관측을 구성해 모델이 예측한 오늘자 목표 비중을
    {ticker: weight}로 반환한다 (합계는 1에서 현금 비중을 뺀 값).
    """
    tickers = meta["tickers"]
    window = meta["window"]
    start = (date.today() - timedelta(days=_INFERENCE_LOOKBACK_DAYS)).isoformat()
    feature_df, _ = load_dataset(tickers, start=start, use_cache=False)
    if len(feature_df) < window:
        raise RuntimeError(f"관측 윈도({window}일)를 채울 데이터가 부족합니다 ({len(feature_df)}일 조회됨).")

    if current_weights is None:
        current_weights = np.array([0.0] * len(tickers) + [1.0], dtype=np.float32)

    macro_window = _macro_window(feature_df, window) if meta.get("macro_enabled") else None
    obs = _build_observation(feature_df, tickers, window, current_weights, macro_window)
    action, _ = model.predict(obs, deterministic=True)
    weights = PortfolioEnv._softmax(action)
    return {ticker: float(w) for ticker, w in zip(tickers, weights[:-1])}


def run(
    client: broker.TossClient, model: Optional[PPO] = None, meta: Optional[Dict] = None
) -> Tuple[List[Dict], Dict]:
    """
    반환값: (종목별 결과 리스트, 포트폴리오 스냅샷). 스냅샷은
    {"portfolio_value": ..., "weights": {ticker: weight, ..., "CASH": weight}}
    형태로, rl/daily_log.py가 실행 기록을 남길 때 사용한다.
    """
    if model is None or meta is None:
        model, meta = load_model_and_meta()

    tickers = meta["tickers"]
    window = meta["window"]
    start = (date.today() - timedelta(days=_INFERENCE_LOOKBACK_DAYS)).isoformat()
    feature_df, close_df = load_dataset(tickers, start=start, use_cache=False)
    if len(feature_df) < window:
        raise RuntimeError(f"관측 윈도({window}일)를 채울 데이터가 부족합니다 ({len(feature_df)}일 조회됨).")

    latest_close = {t: float(close_df[t].iloc[-1]) for t in tickers}
    qty_by_symbol = _holdings_by_symbol(client)
    cash = broker.extract_value(broker.get_buying_power(client, currency="USD"), ["cashBuyingPower"]) or 0.0
    real_current_weights = _current_weights(qty_by_symbol, latest_close, cash, tickers)

    # 드라이런에서는 실계좌가 절대 바뀌지 않아 real_current_weights가 매일
    # "전액 현금"으로 고정된다 - 모델이 매번 같은 시작 상태를 관측해 학습 때와
    # 달리 포지션이 이어지지 않는 원인이었다. 대신 페이퍼 포트폴리오(직전
    # 기록의 목표 비중)를 관측에 넣어 학습 시 환경과 같은 방식으로 상태가
    # 이어지게 한다. 실거래 전환 후에는 실계좌 상태가 그대로 유효하다.
    history = daily_log.load_history()
    prior_value, prior_weights = paper_trading.load_prior_state(history, tickers, initial_capital=cash)
    obs_current_weights = (
        real_current_weights if config.TOSS_LIVE_TRADING else prior_weights.astype(np.float32)
    )

    macro_window = _macro_window(feature_df, window) if meta.get("macro_enabled") else None
    obs = _build_observation(feature_df, tickers, window, obs_current_weights, macro_window)
    action, _ = model.predict(obs, deterministic=True)
    target_weights = PortfolioEnv._softmax(action)

    asset_values = {t: qty_by_symbol.get(t, 0.0) * latest_close[t] for t in tickers}
    total_value = sum(asset_values.values()) + cash
    confirm = config.TOSS_LIVE_TRADING

    results = []
    for i, ticker in enumerate(tickers):
        target_weight = float(target_weights[i])
        diff_value = target_weight * total_value - asset_values[ticker]

        if diff_value > config.RL_REBALANCE_MIN_TRADE_USD:
            order = broker.create_order(
                client, symbol=ticker, side="BUY", order_type="MARKET",
                order_amount=round(diff_value, 2), confirm=confirm,
            )
            results.append({"symbol": ticker, "target_weight": target_weight, "action": "BUY", "order": order})

        elif diff_value < -config.RL_REBALANCE_MIN_TRADE_USD:
            sellable = broker.extract_value(
                broker.get_sellable_quantity(client, ticker),
                ["sellableQuantity", "quantity", "availableQuantity"],
            ) or 0.0
            quantity = min(abs(diff_value) / latest_close[ticker], sellable)
            if quantity <= 0:
                results.append({"symbol": ticker, "target_weight": target_weight, "action": "SKIP_NO_SELLABLE_QTY"})
                continue
            order = broker.create_order(
                client, symbol=ticker, side="SELL", order_type="MARKET",
                quantity=quantity, confirm=confirm,
            )
            results.append({"symbol": ticker, "target_weight": target_weight, "action": "SELL", "order": order})

        else:
            results.append({"symbol": ticker, "target_weight": target_weight, "action": "HOLD"})

    snapshot_weights = {ticker: float(target_weights[i]) for i, ticker in enumerate(tickers)}
    snapshot_weights["CASH"] = float(target_weights[-1])

    if config.TOSS_LIVE_TRADING:
        portfolio_value = total_value
    else:
        prior_date = history[-1]["date"] if history else None
        prior_close = None
        if prior_date is not None:
            prior_ts = pd.Timestamp(prior_date)
            if prior_ts in close_df.index:
                prior_close = close_df.loc[prior_ts, tickers].astype(float).values
        today_close = close_df[tickers].iloc[-1].astype(float).values
        portfolio_value = paper_trading.step(
            prior_value, prior_weights, target_weights.astype(np.float64),
            prior_close, today_close, meta.get("transaction_cost_bps", config.RL_TRANSACTION_COST_BPS),
        )

    snapshot = {"portfolio_value": portfolio_value, "weights": snapshot_weights}

    return results, snapshot
