"""
학습된 PPO 포트폴리오 에이전트로 오늘자 목표 비중을 계산해, 보유 종목을
목표 비중에 맞춰 리밸런싱하는 실거래 연동 모듈. strategy.py(규칙 기반)와는
별개의 경로이며 strategy.py/run_strategy.py는 그대로 둔다 — 둘 중 하나를
선택해서 실행한다.

broker.create_order()의 기존 이중 안전장치(config.TOSS_LIVE_TRADING=false
기본 드라이런, confirm=True 명시 필요)를 strategy.run()과 동일한 방식으로
그대로 사용한다. 이 모듈은 실거래 여부를 직접 판단하지 않는다.
"""
import json
import os
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

import broker
import config
from rl.price_data import load_dataset
from rl.trading_env import PortfolioEnv
from strategy import _extract_value

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


def _build_observation(feature_df, tickers: List[str], window: int, current_weights: np.ndarray) -> np.ndarray:
    n_assets = len(tickers)
    n_features = feature_df.shape[1] // n_assets
    window_slice = feature_df.tail(window).values.astype(np.float32).reshape(window, n_assets, n_features)
    return np.concatenate([window_slice.reshape(-1), current_weights]).astype(np.float32)


def _holdings_by_symbol(client: broker.TossClient) -> Dict[str, float]:
    holdings = broker.get_holdings(client)
    items = holdings.get("holdings", holdings) if isinstance(holdings, dict) else holdings
    if not isinstance(items, list):
        return {}
    qty_by_symbol = {}
    for item in items:
        symbol = item.get("symbol") or item.get("ticker")
        qty = _extract_value(item, ["quantity", "holdingQuantity", "sellableQuantity"])
        if symbol and qty:
            qty_by_symbol[symbol] = qty
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

    obs = _build_observation(feature_df, tickers, window, current_weights)
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
    cash = _extract_value(broker.get_buying_power(client), ["buyingPower", "availableAmount", "amount"]) or 0.0
    current_weights = _current_weights(qty_by_symbol, latest_close, cash, tickers)

    obs = _build_observation(feature_df, tickers, window, current_weights)
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
            sellable = _extract_value(
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
    snapshot = {"portfolio_value": total_value, "weights": snapshot_weights}

    return results, snapshot
