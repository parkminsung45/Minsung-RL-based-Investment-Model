"""
S&P500·NASDAQ100 종목 유니버스 조회. rl/train.py가 RL 학습 종목 유니버스를
가져올 때 쓴다 (과거 뉴스+애널리스트 감성분석 파이프라인은 매크로 블로그
기반 RL로 대체되어 제거됨 - rl/macro_blog.py 참고).
"""
import json
import os
import time
from io import StringIO
from typing import Callable, Dict, List

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# S&P 500 / NASDAQ 100 유니버스
#
# 구성종목은 분기별 리밸런싱으로 바뀌므로, 하드코딩하지 않고 실행 시점마다
# 새로 가져오는 것을 기본으로 한다.
#
# 소스 우선순위:
#   1. stockanalysis.com 종목 리스트 표 (두 지수 모두 지원, 기본)
#   2. (S&P500 한정) 위키피디아 "List of S&P 500 companies" 표
#      - Nasdaq-100 위키피디아 문서는 더 이상 구성종목 표를 포함하지 않아 제외
#   3. 위 소스가 모두 실패하면 마지막으로 성공했던 결과를 로컬 캐시
#      (.cache/universe_cache.json)에서 불러온다.
# ---------------------------------------------------------------------------

_UNIVERSE_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-pipeline/1.0)"}

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
_CACHE_PATH = os.path.join(_CACHE_DIR, "universe_cache.json")

SP500_STOCKANALYSIS_URL = "https://stockanalysis.com/list/sp-500-stocks/"
NASDAQ100_STOCKANALYSIS_URL = "https://stockanalysis.com/list/nasdaq-100-stocks/"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _tickers_from_stockanalysis(url: str) -> List[str]:
    resp = requests.get(url, headers=_UNIVERSE_HEADERS, timeout=15)
    resp.raise_for_status()
    table = pd.read_html(StringIO(resp.text))[0]
    return sorted(set(table["Symbol"].astype(str).str.strip()))


def _sp500_from_wikipedia() -> List[str]:
    resp = requests.get(SP500_WIKI_URL, headers=_UNIVERSE_HEADERS, timeout=15)
    resp.raise_for_status()
    table = pd.read_html(StringIO(resp.text))[0]
    tickers = table["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    return sorted(set(tickers))


def _load_universe_cache() -> Dict:
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_universe_cache(cache: Dict) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _get_universe_tickers(key: str, fetchers: List[Callable[[], List[str]]]) -> List[str]:
    for fetch in fetchers:
        try:
            tickers = fetch()
            if tickers:
                cache = _load_universe_cache()
                cache[key] = {
                    "tickers": tickers,
                    "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                _save_universe_cache(cache)
                return tickers
        except Exception as e:
            print(f"[universe] {key} 조회 실패 ({fetch!r}): {e}")

    cache = _load_universe_cache()
    cached = cache.get(key)
    if cached:
        print(f"[universe] 실시간 조회 실패, 캐시된 {key} 목록 사용 (기준: {cached['fetched_at']})")
        return cached["tickers"]

    raise RuntimeError(f"{key} 종목 리스트를 가져오지 못했고, 캐시도 없습니다.")


def get_sp500_tickers() -> List[str]:
    return _get_universe_tickers(
        "sp500",
        [lambda: _tickers_from_stockanalysis(SP500_STOCKANALYSIS_URL), _sp500_from_wikipedia],
    )


def get_nasdaq100_tickers() -> List[str]:
    return _get_universe_tickers(
        "nasdaq100",
        [lambda: _tickers_from_stockanalysis(NASDAQ100_STOCKANALYSIS_URL)],
    )


def get_universe_tickers() -> List[str]:
    """S&P 500과 NASDAQ 100의 합집합(중복 제거, 정렬)."""
    return sorted(set(get_sp500_tickers()) | set(get_nasdaq100_tickers()))
