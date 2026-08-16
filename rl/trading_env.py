"""
일 단위 포트폴리오 배분 강화학습 환경 (gymnasium).

자산 = tickers + 현금. 매 스텝 에이전트가 목표 비중(softmax 정규화, 합=1,
공매도 없음)을 정하면 그 비중으로 리밸런싱(회전율만큼 거래비용 차감)한 뒤,
다음 거래일 종가로 포트폴리오 가치를 갱신한다. reward = 일간 로그수익률.
"""
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_df: pd.DataFrame,
        close_df: pd.DataFrame,
        tickers: List[str],
        window: int = 20,
        episode_length: int = 126,
        transaction_cost_bps: float = 10.0,
        random_start: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if list(feature_df.index) != list(close_df.index):
            raise ValueError("feature_df와 close_df의 날짜 인덱스가 일치해야 합니다.")
        if len(feature_df) <= window + episode_length:
            raise ValueError(
                f"데이터가 부족합니다 (일수={len(feature_df)}, "
                f"필요=window({window})+episode_length({episode_length})+1 이상)."
            )

        self.feature_df = feature_df
        self.close_df = close_df
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.n_features = feature_df.shape[1] // self.n_assets
        self.window = window
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost_bps / 10_000
        self.random_start = random_start

        obs_dim = self.window * self.n_assets * self.n_features + (self.n_assets + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # n_assets개 종목 + 현금 1개 = n_assets+1 로짓. step()에서 softmax로 정규화한다.
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.n_assets + 1,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._feature_values = feature_df.values.astype(np.float32).reshape(
            len(feature_df), self.n_assets, self.n_features
        )
        self._close_values = close_df[tickers].values.astype(np.float64)

        self._step_idx = self.window
        self._end_idx = len(feature_df) - 1
        self._weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self._weights[-1] = 1.0
        self._portfolio_value = 1.0

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.max()
        e = np.exp(x)
        return (e / e.sum()).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        window_slice = self._feature_values[self._step_idx - self.window : self._step_idx]
        return np.concatenate([window_slice.reshape(-1), self._weights]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        last_valid_start = len(self.feature_df) - self.episode_length - 1
        if self.random_start and last_valid_start > self.window:
            self._step_idx = int(self._rng.integers(self.window, last_valid_start))
        else:
            self._step_idx = self.window

        self._end_idx = min(self._step_idx + self.episode_length, len(self.feature_df) - 1)
        self._weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self._weights[-1] = 1.0  # 에피소드는 전량 현금에서 시작
        self._portfolio_value = 1.0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        weights = self._softmax(action)
        turnover = float(np.abs(weights - self._weights).sum())
        cost = turnover * self.transaction_cost
        self._weights = weights

        idx = self._step_idx
        next_idx = idx + 1
        asset_returns = (self._close_values[next_idx] / self._close_values[idx]) - 1.0
        portfolio_return = float(np.dot(weights[:-1], asset_returns)) - cost  # 현금 수익률 0

        self._portfolio_value *= 1.0 + portfolio_return
        reward = float(np.log(max(1.0 + portfolio_return, 1e-6)))

        self._step_idx = next_idx
        terminated = self._step_idx >= self._end_idx
        truncated = False
        info = {
            "portfolio_value": self._portfolio_value,
            "weights": weights.copy(),
            "date": self.feature_df.index[idx],
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass
