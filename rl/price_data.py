"""
과거 가격(OHLCV) 조회 및 RL 상태(state)용 기술적 지표 계산.

뉴스/애널리스트 신호(data_pipeline.py)는 무료 API 티어의 호출 한도 때문에
수년치 일별 히스토리를 구할 수 없다. 그래서 RL 학습·백테스트용 상태는
yfinance로 무료·무제한 조회 가능한 가격 데이터 기반 기술적 지표만으로 구성한다.
"""
import hashlib
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")

FEATURE_COLUMNS = [
    "return_1d",
    "ma5_return",
    "ma20_return",
    "ma60_return",
    "volatility_20d",
    "rsi_14",
    "volume_zscore_20d",
]


def _cache_path(tickers: List[str], start: str, end: Optional[str]) -> str:
    key = f"{','.join(sorted(tickers))}|{start}|{end or 'latest'}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(_CACHE_DIR, f"price_history_{digest}.pkl")


def fetch_price_history(
    tickers: List[str], start: str, end: Optional[str] = None, use_cache: bool = True
) -> pd.DataFrame:
    """
    tickers의 일별 OHLCV를 조회한다 (배당/분할 반영된 auto-adjust 가격).
    반환 DataFrame: (날짜) 인덱스 x (ticker, field) MultiIndex 컬럼
    (field: Open/High/Low/Close/Volume).
    `.cache/price_history_*.pkl`에 캐싱해 동일 조건 재실행 시 재조회하지 않는다.
    """
    path = _cache_path(tickers, start, end)
    if use_cache and os.path.exists(path):
        return pd.read_pickle(path)

    data = yf.download(
        tickers, start=start, end=end, group_by="ticker", auto_adjust=True, progress=False
    )
    if not isinstance(data.columns, pd.MultiIndex):
        # 티커가 하나뿐이면 yfinance가 flat 컬럼을 반환하므로 형식을 맞춘다.
        data.columns = pd.MultiIndex.from_product([tickers, data.columns])

    data = data.dropna(how="all")

    if use_cache:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        data.to_pickle(path)

    return data


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.astype(float).fillna(50.0)  # 데이터 부족/무변동 구간은 중립값(50)


def compute_features(price_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    price_df: fetch_price_history()가 반환한 (ticker, field) MultiIndex 컬럼 DataFrame.
    반환: (날짜) 인덱스 x (ticker, feature) MultiIndex 컬럼. FEATURE_COLUMNS 참고.
    롤링 윈도(최대 60일) 워밍업 구간의 NaN 행은 제거한다.
    """
    feature_frames = {}
    for ticker in tickers:
        close = price_df[(ticker, "Close")]
        volume = price_df[(ticker, "Volume")]
        daily_return = close.pct_change()
        volume_mean = volume.rolling(20).mean()
        volume_std = volume.rolling(20).std()

        feature_frames[ticker] = pd.DataFrame({
            "return_1d": daily_return,
            "ma5_return": daily_return.rolling(5).mean(),
            "ma20_return": daily_return.rolling(20).mean(),
            "ma60_return": daily_return.rolling(60).mean(),
            "volatility_20d": daily_return.rolling(20).std(),
            "rsi_14": _rsi(close, 14),
            "volume_zscore_20d": (volume - volume_mean) / volume_std.replace(0, np.nan),
        })

    combined = pd.concat(feature_frames, axis=1)
    return combined.dropna(how="any")


def load_dataset(
    tickers: List[str], start: str, end: Optional[str] = None, use_cache: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    학습/백테스트/실거래 추론에서 공통으로 쓰는 진입점.
    반환: (feature_df, close_df) — 둘 다 같은 날짜 인덱스로 정렬됨.
      feature_df: (ticker, feature) MultiIndex 컬럼, compute_features() 결과.
      close_df: ticker별 종가 컬럼 (환경의 포트폴리오 가치 갱신에 사용).
    """
    price_df = fetch_price_history(tickers, start, end, use_cache=use_cache)
    feature_df = compute_features(price_df, tickers)
    close_df = pd.DataFrame({ticker: price_df[(ticker, "Close")] for ticker in tickers})
    close_df = close_df.loc[feature_df.index]
    return feature_df, close_df


def load_dataset_filtered(
    tickers: List[str], start: str, end: Optional[str] = None, use_cache: bool = True,
    min_coverage: float = 0.98,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """
    S&P 500 같은 큰 유니버스로 학습할 때 쓰는 진입점. compute_features()의
    dropna(how="any")는 한 종목이라도 구간 일부에 결측(최근 상장·조회 실패 등)이
    있으면 그 날짜 전체가 통째로 사라진다 - 500종목 중 하나만 상장 이력이
    짧아도 전체 학습 구간이 초토화될 수 있다. 그래서 종목별 종가 결측 비율을
    먼저 확인해, 구간 전체를 min_coverage 이상 커버하는 종목만 남긴 뒤
    feature_df/close_df를 만든다.
    반환: (실제로 쓰인 티커 리스트, feature_df, close_df).
    """
    price_df = fetch_price_history(tickers, start, end, use_cache=use_cache)
    total_days = len(price_df)
    kept, dropped = [], []
    for ticker in tickers:
        try:
            coverage = price_df[(ticker, "Close")].notna().sum() / total_days if total_days else 0.0
        except KeyError:
            coverage = 0.0
        (kept if coverage >= min_coverage else dropped).append(ticker)

    if dropped:
        preview = ", ".join(dropped[:15]) + (" ..." if len(dropped) > 15 else "")
        print(f"[price_data] 상장 이력 부족/조회 실패로 제외된 종목 {len(dropped)}개: {preview}")

    feature_df = compute_features(price_df, kept)
    close_df = pd.DataFrame({ticker: price_df[(ticker, "Close")] for ticker in kept})
    close_df = close_df.loc[feature_df.index]
    return kept, feature_df, close_df
