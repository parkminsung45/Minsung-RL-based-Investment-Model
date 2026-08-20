"""
"헨리의 퀀트대학"(blog.naver.com/leebisu) 블로그의 "글로벌 매크로 트렌드"
연재글을 매일 크롤링해 KR-FinBert-SC로 감성 점수화하고, RL 관측(observation)에
쓸 날짜별 매크로 시그널로 저장한다. 기존 뉴스(Alpha Vantage)+애널리스트
컨센서스(Finnhub) 감성분석 전략을 대체한다.

구성:
  - fetch_post_list(): 네이버 블로그 내부 위젯 API(PostTitleListAsync.naver)로
    카테고리("Global Macro", categoryNo=29) 전체 글 목록(logNo, 날짜)을 가져온다.
    공식 API가 아니라 블로그 자체 JS 위젯이 쓰는 엔드포인트라 언제든 바뀔 수 있음.
  - fetch_post_text(): 모바일 블로그 페이지(m.blog.naver.com)에서 본문
    (.se-main-container)만 추출한다. PC 페이지는 frameset이라 파싱이 더 번거로움.
  - score_text(): 본문을 불릿("•") 단위로 쪼개 KR-FinBert-SC로 각각 점수화
    (score = P(positive) - P(negative), -1~1)한 뒤 평균한다.
  - rl/macro_daily_scores.json: 날짜->점수 저장소. 학습 시 과거 전체를
    한 번만 크롤링/점수화해 채워두고(get_macro_series), 이후에는
    daily_update()가 그날 새 글만 추가한다 (10시 launchd 잡에서 호출).

실행: python -m rl.macro_blog            # 오늘자 글 점수화 + 저장
      python -m rl.macro_blog --push     # 저장 후 git commit/push까지
      python -m rl.macro_blog --backfill # 카테고리 전체 과거 글 채우기(최초 1회용)
"""
import argparse
import json
import os
import re
import subprocess
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup

BLOG_ID = "leebisu"
CATEGORY_NO = 29  # "Global Macro" 카테고리
LIST_URL = "https://blog.naver.com/PostTitleListAsync.naver"
MOBILE_POST_URL = "https://m.blog.naver.com/{blog_id}/{log_no}"
SENTIMENT_MODEL = "snunlp/KR-FinBert-SC"

_HEADERS = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15"}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_REPO_ROOT, ".cache", "macro_blog")
_SCORES_PATH = os.path.join(_REPO_ROOT, "rl", "macro_daily_scores.json")

_REQUEST_DELAY_SEC = 0.4

_sentiment_pipeline = None


def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        _sentiment_pipeline = pipeline("text-classification", model=SENTIMENT_MODEL, top_k=None, truncation=True)
    return _sentiment_pipeline


def _parse_add_date(raw: str, fetched_at: date) -> str:
    """PostTitleListAsync.naver의 addDate는 오늘 글은 '47분 전'/'3시간 전' 같은
    상대 시각으로, 그 외에는 '2026. 8. 21.' 같은 절대 날짜로 온다."""
    raw = raw.strip()
    if raw.endswith("전"):
        return fetched_at.isoformat()
    m = re.match(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?", raw)
    if not m:
        raise ValueError(f"addDate 형식을 해석할 수 없습니다: {raw!r}")
    year, month, day = (int(x) for x in m.groups())
    return date(year, month, day).isoformat()


def fetch_post_list(max_pages: int = 40, count_per_page: int = 30) -> List[Tuple[str, str]]:
    """(logNo, ISO 날짜) 리스트를 최신순으로 반환한다. API가 범위를 벗어난
    페이지 요청에는 마지막 유효 페이지를 그대로 반복 응답하므로, 직전
    페이지와 내용이 같아지면 멈춘다."""
    today = date.today()
    seen_logs = set()
    results = []
    prev_page_logs = None

    for page in range(1, max_pages + 1):
        resp = requests.get(
            LIST_URL,
            params={"blogId": BLOG_ID, "categoryNo": CATEGORY_NO, "countPerPage": count_per_page, "currentPage": page},
            headers=_HEADERS, timeout=15,
        )
        resp.raise_for_status()
        text = resp.text
        log_nos = re.findall(r'"logNo":"(\d+)"', text)
        add_dates = re.findall(r'"addDate":"([^"]*)"', text)
        if not log_nos:
            break
        if log_nos == prev_page_logs:
            break
        prev_page_logs = log_nos

        for log_no, add_date_raw in zip(log_nos, add_dates):
            if log_no in seen_logs:
                continue
            seen_logs.add(log_no)
            results.append((log_no, _parse_add_date(add_date_raw, today)))

        if len(log_nos) < count_per_page:
            break
        time.sleep(_REQUEST_DELAY_SEC)

    return results


def fetch_post_text(log_no: str, use_cache: bool = True) -> str:
    """모바일 블로그 페이지에서 본문 텍스트만 추출한다. .cache/macro_blog/에
    로컬 캐싱해 같은 글을 반복 스코어링할 때 다시 긁지 않는다."""
    cache_path = os.path.join(_CACHE_DIR, f"{log_no}.txt")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return f.read()

    resp = requests.get(MOBILE_POST_URL.format(blog_id=BLOG_ID, log_no=log_no), headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    container = soup.select_one(".se-main-container")
    text = container.get_text("\n", strip=True) if container else ""

    if use_cache and text:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)

    return text


_BULLET_PREFIXES = ("•", "✅", "▶", "○", "■", "-", "*")
_MIN_BULLET_LEN = 20  # 섹션 헤더("🇰🇷 한국 주식")나 출처 표기("GS", "Flow Show") 같은
                       # 짧은 줄을 걸러내기 위한 최소 길이. 블로그가 시기별로
                       # 불릿 기호를 •에서 ✅ 등으로 바꿔왔기 때문에, 특정 기호
                       # 하나에만 의존하지 않고 "충분히 긴 줄"을 실질 내용으로 본다.


def score_text(text: str) -> Optional[float]:
    """본문을 문장 단위로 쪼개 KR-FinBert-SC 점수(P(positive)-P(negative))를
    평균한다. 유효한 문장이 하나도 없으면(파싱 실패 등) None을 반환한다."""
    bullets = []
    for raw_line in text.split("\n"):
        line = raw_line.strip().lstrip("​").strip()
        for prefix in _BULLET_PREFIXES:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if len(line) >= _MIN_BULLET_LEN:
            bullets.append(line)
    if not bullets:
        return None

    pipe = _get_sentiment_pipeline()
    scores = []
    for bullet in bullets:
        result = {r["label"]: r["score"] for r in pipe(bullet)[0]}
        scores.append(result.get("positive", 0.0) - result.get("negative", 0.0))
    return sum(scores) / len(scores)


def load_daily_scores() -> Dict[str, float]:
    if not os.path.exists(_SCORES_PATH):
        return {}
    with open(_SCORES_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_daily_scores(scores: Dict[str, float]) -> None:
    with open(_SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(scores.items())), f, ensure_ascii=False, indent=2)
        f.write("\n")


def backfill(start_date: Optional[str] = None) -> Dict[str, float]:
    """카테고리 전체 글 목록을 가져와, rl/macro_daily_scores.json에 아직 없는
    날짜만 새로 점수화해 채운다. 같은 날짜에 글이 여러 건이면 평균한다.
    학습 시(또는 최초 1회) 과거분을 한꺼번에 채우는 용도 - 매일 반복 실행해도
    이미 있는 날짜는 다시 크롤링하지 않는다."""
    scores = load_daily_scores()
    posts = fetch_post_list()
    if start_date:
        posts = [(log_no, d) for log_no, d in posts if d >= start_date]

    by_date: Dict[str, List[float]] = {}
    for log_no, post_date in posts:
        if post_date in scores:
            continue
        text = fetch_post_text(log_no)
        score = score_text(text)
        if score is not None:
            by_date.setdefault(post_date, []).append(score)
        time.sleep(_REQUEST_DELAY_SEC)

    for d, vals in by_date.items():
        scores[d] = sum(vals) / len(vals)

    _save_daily_scores(scores)
    return scores


def daily_update(today: Optional[date] = None) -> Optional[float]:
    """10시 launchd 잡에서 호출 - 오늘자 글만 확인해 점수화하고 저장소에 추가한다.
    오늘 글이 아직 없으면(포스팅 지연 등) None을 반환하고 저장소는 건드리지 않는다."""
    today = today or date.today()
    today_iso = today.isoformat()

    posts = fetch_post_list(max_pages=2)
    today_logs = [log_no for log_no, d in posts if d == today_iso]
    if not today_logs:
        print(f"[macro_blog] {today_iso} 자 글이 아직 없습니다 - 건너뜁니다.")
        return None

    day_scores = []
    for log_no in today_logs:
        text = fetch_post_text(log_no)
        score = score_text(text)
        if score is not None:
            day_scores.append(score)

    if not day_scores:
        print(f"[macro_blog] {today_iso} 자 글에서 불릿을 찾지 못해 점수화 실패.")
        return None

    final_score = sum(day_scores) / len(day_scores)
    scores = load_daily_scores()
    scores[today_iso] = round(final_score, 4)
    _save_daily_scores(scores)
    print(f"[macro_blog] {today_iso} 매크로 점수 저장: {final_score:+.4f}")
    return final_score


def get_macro_series(start_date: str, end_date: Optional[str] = None, backfill_missing: bool = True) -> pd.Series:
    """[start_date, end_date] 구간의 날짜별 매크로 점수를 pd.Series(날짜 인덱스)로
    반환한다. 저장소에 없는 과거 구간은 backfill_missing=True일 때 크롤링해 채운다.
    end_date 생략 시 저장소에 있는 가장 최근 날짜까지."""
    scores = backfill(start_date) if backfill_missing else load_daily_scores()
    items = [(d, v) for d, v in scores.items() if d >= start_date and (end_date is None or d <= end_date)]
    if not items:
        return pd.Series(dtype=float)
    items.sort()
    dates, values = zip(*items)
    return pd.Series(values, index=pd.to_datetime(dates))


def push_scores() -> None:
    """rl/macro_daily_scores.json만 커밋/푸시한다 (다른 변경사항은 건드리지 않음)."""
    path = os.path.join("rl", "macro_daily_scores.json")
    try:
        subprocess.run(["git", "add", path], cwd=_REPO_ROOT, check=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet", "--", path], cwd=_REPO_ROOT)
        if diff.returncode == 0:
            print("[macro_blog] 변경 사항 없음 - 커밋/푸시 생략")
            return
        subprocess.run(
            ["git", "commit", "-m", f"매크로 블로그 점수: {date.today().isoformat()}", "--", path],
            cwd=_REPO_ROOT, check=True,
        )
        subprocess.run(["git", "push"], cwd=_REPO_ROOT, check=True)
        print("[macro_blog] 커밋/푸시 완료")
    except subprocess.CalledProcessError as exc:
        print(f"[macro_blog] 경고: git 커밋/푸시 실패 ({exc})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="저장 후 git commit/push")
    parser.add_argument("--backfill", action="store_true", help="카테고리 전체 과거 글을 채운다 (최초 1회용)")
    args = parser.parse_args()

    if args.backfill:
        result = backfill()
        print(f"[macro_blog] 저장소 총 {len(result)}개 날짜")
    else:
        daily_update()

    if args.push:
        push_scores()
