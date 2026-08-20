"""
전역 설정 파일.
API 키는 .env 파일에서 불러온다 (.env.example 참고).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 토스증권 Open API (실거래 연동, 샌드박스 없음 - broker.py 참고)
TOSS_CLIENT_ID = os.getenv("TOSS_CLIENT_ID", "")
TOSS_CLIENT_SECRET = os.getenv("TOSS_CLIENT_SECRET", "")

# 반드시 명시적으로 "true"를 설정해야 실제 주문이 나간다. 기본값은 항상 드라이런.
TOSS_LIVE_TRADING = os.getenv("TOSS_LIVE_TRADING", "false").strip().lower() == "true"

# 결과 저장 경로
OUTPUT_DIR = "output"

# --- 강화학습 기반 포트폴리오 에이전트 (rl/) ---
# state에 포함할 과거 일수 (기술적 지표 롤링 윈도)
RL_LOOKBACK_WINDOW_DAYS = 20
# 리밸런싱 시 비중 회전율(turnover)에 부과하는 거래비용 (bps, 1bp = 0.01%)
RL_TRANSACTION_COST_BPS = 10.0
# yfinance로 내려받을 과거 가격 데이터 기간 (년). 이 중 RL_TRAIN_TEST_SPLIT
# 비율만큼 학습에 쓰고 나머지 최근 구간은 백테스트/드리프트 점검용으로 보류.
RL_TRAIN_YEARS = 6
RL_TRAIN_TEST_SPLIT = 0.8
# 학습 에피소드 길이 (거래일). 학습 구간 내 랜덤 시작일에서 이 길이만큼 진행.
# 매크로 블로그 시그널(rl/macro_blog.py) 도입으로 학습 구간이 블로그 시작일
# (2025-11-09) 이후로 짧아져(약 150~190거래일), 기존 126일짜리 에피소드는
# random_start가 사실상 고정 시작점 근처로만 몰려 다양성이 거의 없어진다.
# 40일로 줄여 랜덤 시작 범위를 넓힌다.
RL_EPISODE_LENGTH_DAYS = 40
RL_MODEL_PATH = "rl/models/ppo_portfolio.zip"
# 리밸런싱 시 이 금액(달러) 미만의 목표-현재 비중 차이는 무시한다 (잦은 소액 거래 방지)
RL_REBALANCE_MIN_TRADE_USD = 5.0
# daily_run.py 드리프트 점검: 보류 구간 백테스트 최대낙폭(MDD)이 이 값을
# 넘으면 이상 상황으로 보고 리밸런싱 없이 종료한다.
RL_DRIFT_MAX_DRAWDOWN = 0.35
# 드라이런(TOSS_LIVE_TRADING=false) 동안 rl/paper_trading.py가 굴리는 가상
# 포트폴리오의 시작 자본(달러). 실계좌와 무관 - 대시보드용 시뮬레이션 값이다.
RL_PAPER_INITIAL_CAPITAL = 10_000.0
