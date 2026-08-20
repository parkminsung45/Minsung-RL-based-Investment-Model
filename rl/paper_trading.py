"""
드라이런(TOSS_LIVE_TRADING=false) 동안 대시보드에 남길 포트폴리오 가치를
실제 토스 계좌 상태와 무관하게 시뮬레이션한다.

기존 문제: 드라이런에서는 broker.create_order()가 실제로 주문을 보내지
않으므로 실계좌 보유 종목/현금이 절대 바뀌지 않는다. rl_strategy.run()이
그 실계좌 상태로 portfolio_value를 계산해 기록했기 때문에
daily_run_history.json의 가치가 매일 같은 숫자로 고정됐다(일간 수익률 0%).

이 모듈은 rl/daily_run_history.json의 직전 기록(가치 + 비중)만으로 상태를
이어받아, rl/trading_env.py가 학습 때 쓰는 것과 동일한 갱신식
(가치 *= 1 + 목표비중·자산수익률 - 회전율*거래비용)으로 오늘의 가치를 굴린다.
실거래 주문 계산(diff_value 등)은 여전히 실계좌 잔고를 기준으로 하므로
실거래 전환 시 그대로 유효하다 - 이 모듈은 로깅/대시보드용 가치만 대체한다.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

import config


def load_prior_state(history: List[Dict], tickers: List[str]) -> Tuple[float, np.ndarray]:
    """직전 기록에서 (포트폴리오 가치, [종목별 비중..., 현금비중]) 을 복원한다.
    기록이 없으면 RL_PAPER_INITIAL_CAPITAL 전액 현금에서 새로 시작한다."""
    if not history:
        weights = np.array([0.0] * len(tickers) + [1.0], dtype=np.float64)
        return config.RL_PAPER_INITIAL_CAPITAL, weights

    last = history[-1]
    weights = np.array(
        [last["weights"].get(t, 0.0) for t in tickers] + [last["weights"].get("CASH", 0.0)],
        dtype=np.float64,
    )
    return float(last["portfolio_value"]), weights


def step(
    prior_value: float,
    prior_weights: np.ndarray,
    target_weights: np.ndarray,
    prior_close: Optional[np.ndarray],
    today_close: np.ndarray,
    transaction_cost_bps: float,
) -> float:
    """전날 비중이 오늘 종가로 이동한 만큼 가치를 갱신한 뒤, 오늘의 목표
    비중으로 리밸런싱하는 회전율 비용을 뺀 새 포트폴리오 가치를 반환한다.
    prior_close가 없으면(첫 실행 등) 시장 변동 없이 초기 가치를 그대로 쓴다."""
    if prior_close is None:
        asset_return = 0.0
    else:
        asset_returns = today_close / prior_close - 1.0
        asset_return = float(np.dot(prior_weights[:-1], asset_returns))

    turnover = float(np.abs(target_weights - prior_weights).sum())
    cost = turnover * (transaction_cost_bps / 10_000)
    return prior_value * (1.0 + asset_return - cost)
