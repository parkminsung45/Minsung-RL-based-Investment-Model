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

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import config
import data_pipeline
from rl import macro_blog
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


def build_env(feature_df, close_df, tickers: list, random_start: bool, macro_series=None) -> PortfolioEnv:
    return PortfolioEnv(
        feature_df,
        close_df,
        tickers=tickers,
        window=config.RL_LOOKBACK_WINDOW_DAYS,
        episode_length=config.RL_EPISODE_LENGTH_DAYS,
        transaction_cost_bps=config.RL_TRANSACTION_COST_BPS,
        random_start=random_start,
        macro_series=macro_series,
    )


def run(timesteps: int, model_path: str = None) -> None:
    model_path = model_path or config.RL_MODEL_PATH

    # RL 학습 종목 유니버스는 S&P 500 전체를 쓴다 -
    # data_pipeline.get_sp500_tickers()가 실시간 조회(실패 시 로컬 캐시)로 가져온다.
    # stockanalysis.com 표는 BRK.B처럼 점(.) 표기를 쓰는데 yfinance는 하이픈
    # (BRK-B)을 기대하므로 다운로드 전에 변환한다.
    candidates = [t.replace(".", "-") for t in data_pipeline.get_sp500_tickers()]

    # 매크로 블로그(rl/macro_blog.py) 시그널을 관측에 포함하므로, 학습 구간은
    # RL_TRAIN_YEARS(가격 데이터 기준 6년)가 아니라 블로그 히스토리가 존재하는
    # 시점부터로 맞춘다 - 그 이전 구간은 매크로 시그널이 아예 없어 학습에 넣어도
    # 의미가 없다. get_macro_series()가 카테고리 전체 글을 크롤링/점수화해
    # rl/macro_daily_scores.json에 채운다(이미 채워져 있으면 새 글만 추가).
    print("[1/4] 매크로 블로그(글로벌 매크로 트렌드) 히스토리 확인 중...")
    macro_series_full = macro_blog.get_macro_series(start_date="2000-01-01", backfill_missing=True)
    if macro_series_full.empty:
        raise RuntimeError("매크로 블로그 히스토리를 하나도 가져오지 못했습니다 - 크롤링 실패 가능성.")
    start = macro_series_full.index.min().date().isoformat()
    print(f"      매크로 데이터 {len(macro_series_full)}개 날짜, 시작일 {start} - 학습 구간을 이 시점부터로 맞춘다")

    print(f"[2/4] 가격 데이터 로드 중... S&P500 후보 {len(candidates)}개, 시작일: {start}")
    tickers, feature_df, close_df = load_dataset_filtered(candidates, start=start)
    print(
        f"      매크로 데이터가 있는 전체 기간 중 데이터가 있는 {len(tickers)}개 종목 사용, "
        f"{len(feature_df)}개 거래일 로드 완료 ({feature_df.index[0].date()} ~ {feature_df.index[-1].date()})"
    )

    train_feature_df, train_close_df, holdout_feature_df, holdout_close_df = split_dataset(
        feature_df, close_df, config.RL_LOOKBACK_WINDOW_DAYS, config.RL_TRAIN_TEST_SPLIT
    )
    print(f"      학습 {len(train_feature_df)}일 / 보류(holdout) {len(holdout_feature_df)}일")

    print(f"[3/4] PPO 학습 중... timesteps={timesteps}")
    env = Monitor(build_env(train_feature_df, train_close_df, tickers, random_start=True, macro_series=macro_series_full))
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)

    print("[4/4] 모델 저장 중...")
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
        "macro_enabled": True,
        "macro_source": "leebisu_global_macro_krfinbert",
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
