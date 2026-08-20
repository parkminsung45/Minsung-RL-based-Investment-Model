"""
PPO 기반 포트폴리오 배분 에이전트 학습.

실행 방법:
    python -m rl.train
    python -m rl.train --timesteps 20000   # 스모크 테스트용 소량 학습

가격 데이터(yfinance)를 시간 순서대로 학습/보류(holdout) 구간으로 나누고
(뒤섞지 않음 — 미래 데이터 누수 방지), 학습 구간에서만 PPO를 학습시킨다.
보류 구간은 rl/backtest.py와 rl/daily_run.py의 드리프트 점검에서 사용한다.
"""
import argparse
import json
import os
from datetime import date, timedelta

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import config
import data_pipeline
from rl.price_data import load_dataset_filtered
from rl.trading_env import PortfolioEnv


def _meta_path(model_path: str) -> str:
    return os.path.splitext(model_path)[0] + ".meta.json"


def split_dataset(feature_df, close_df, window: int, split_ratio: float):
    """
    시간 순서를 지키는 train/holdout 분할. holdout 쪽은 자기 환경의 첫 관측
    윈도를 채울 수 있도록 분할 지점 이전 `window`일을 겹쳐서 포함한다.
    """
    split_idx = int(len(feature_df) * split_ratio)
    train_feature_df = feature_df.iloc[:split_idx]
    train_close_df = close_df.iloc[:split_idx]

    holdout_start = max(split_idx - window, 0)
    holdout_feature_df = feature_df.iloc[holdout_start:]
    holdout_close_df = close_df.iloc[holdout_start:]

    return train_feature_df, train_close_df, holdout_feature_df, holdout_close_df


def build_env(feature_df, close_df, tickers: list, random_start: bool) -> PortfolioEnv:
    return PortfolioEnv(
        feature_df,
        close_df,
        tickers=tickers,
        window=config.RL_LOOKBACK_WINDOW_DAYS,
        episode_length=config.RL_EPISODE_LENGTH_DAYS,
        transaction_cost_bps=config.RL_TRANSACTION_COST_BPS,
        random_start=random_start,
    )


def run(timesteps: int, model_path: str = None) -> None:
    model_path = model_path or config.RL_MODEL_PATH

    # RL 학습 종목 유니버스는 config.WATCHLIST(main.py 뉴스+애널리스트 파이프라인용
    # 소수 관심종목, Alpha Vantage 무료 티어 제약)와 별개로 S&P 500 전체를 쓴다 -
    # data_pipeline.get_sp500_tickers()가 실시간 조회(실패 시 로컬 캐시)로 가져온다.
    # stockanalysis.com 표는 BRK.B처럼 점(.) 표기를 쓰는데 yfinance는 하이픈
    # (BRK-B)을 기대하므로 다운로드 전에 변환한다.
    candidates = [t.replace(".", "-") for t in data_pipeline.get_sp500_tickers()]
    start = (date.today() - timedelta(days=int(365.25 * config.RL_TRAIN_YEARS))).isoformat()
    print(f"[1/3] 가격 데이터 로드 중... S&P500 후보 {len(candidates)}개, 시작일: {start}")
    tickers, feature_df, close_df = load_dataset_filtered(candidates, start=start)
    print(
        f"      학습 기간({config.RL_TRAIN_YEARS}년) 전체 데이터가 있는 {len(tickers)}개 종목 사용, "
        f"{len(feature_df)}개 거래일 로드 완료 ({feature_df.index[0].date()} ~ {feature_df.index[-1].date()})"
    )

    train_feature_df, train_close_df, holdout_feature_df, holdout_close_df = split_dataset(
        feature_df, close_df, config.RL_LOOKBACK_WINDOW_DAYS, config.RL_TRAIN_TEST_SPLIT
    )
    print(f"      학습 {len(train_feature_df)}일 / 보류(holdout) {len(holdout_feature_df)}일")

    print(f"[2/3] PPO 학습 중... timesteps={timesteps}")
    env = Monitor(build_env(train_feature_df, train_close_df, tickers, random_start=True))
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)

    print("[3/3] 모델 저장 중...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    meta = {
        "tickers": tickers,
        "window": config.RL_LOOKBACK_WINDOW_DAYS,
        "episode_length": config.RL_EPISODE_LENGTH_DAYS,
        "transaction_cost_bps": config.RL_TRANSACTION_COST_BPS,
        "train_start": feature_df.index[0].date().isoformat(),
        "split_date": feature_df.index[len(train_feature_df) - 1].date().isoformat(),
        "data_end": feature_df.index[-1].date().isoformat(),
    }
    with open(_meta_path(model_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"완료: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()
    run(args.timesteps, args.model_path)
