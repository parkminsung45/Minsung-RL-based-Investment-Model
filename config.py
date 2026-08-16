"""
전역 설정 파일.
API 키는 .env 파일에서 불러온다 (.env.example 참고).
"""
import os
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# 토스증권 Open API (실거래 연동, 샌드박스 없음 - broker/orders.py 참고)
TOSS_CLIENT_ID = os.getenv("TOSS_CLIENT_ID", "")
TOSS_CLIENT_SECRET = os.getenv("TOSS_CLIENT_SECRET", "")

# 반드시 명시적으로 "true"를 설정해야 실제 주문이 나간다. 기본값은 항상 드라이런.
TOSS_LIVE_TRADING = os.getenv("TOSS_LIVE_TRADING", "false").strip().lower() == "true"

# 뉴스+애널리스트 결합 시그널(main.py)을 계산할 소수 관심종목.
# Alpha Vantage 무료 티어(하루 25회)로 감당 가능한 규모로 유지할 것.
WATCHLIST = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

# 결과 저장 경로
OUTPUT_DIR = "output"

# 신호 결합 가중치: NEWS_WEIGHT + ANALYST_WEIGHT = 1 이어야 함
NEWS_WEIGHT = 0.5
ANALYST_WEIGHT = 0.5

# Alpha Vantage 무료 티어는 분당 호출 제한이 있어 티커 간 대기 시간(초) 필요
NEWS_FETCH_DELAY_SEC = 12.0

# scan_universe.py: S&P500+NASDAQ100 전체 스캔 시 Finnhub 호출 간 대기시간(초).
# 무료 티어 분당 60회 한도에 대응.
FINNHUB_UNIVERSE_DELAY_SEC = 1.0

# 매매 전략 (strategy.py). score(-1~1) 기준:
#   score > BUY_THRESHOLD  -> 매수
#   score < SELL_THRESHOLD -> 매도 (보유 중일 때만)
#   그 외                   -> 홀드
BUY_THRESHOLD = 0.3
SELL_THRESHOLD = -0.3

# 매수 시 종목당 매수가능금액(buying power) 대비 매수 비율
POSITION_SIZE_PCT = 0.05

# 재무 건전성 필터 (data_pipeline.passes_financial_health). Finnhub Basic
# Financials(stock/metric) 기준: 순이익률>0, ROE>0, 부채비율(D/E)<2.0.
# 매수 후보 종목이 이 조건을 통과하지 못하면 매수하지 않는다.
MIN_NET_MARGIN = 0.0
MIN_ROE = 0.0
MAX_DEBT_TO_EQUITY = 2.0

# --- 강화학습 기반 포트폴리오 에이전트 (rl/, 실험적) ---
# state에 포함할 과거 일수 (기술적 지표 롤링 윈도)
RL_LOOKBACK_WINDOW_DAYS = 20
# 리밸런싱 시 비중 회전율(turnover)에 부과하는 거래비용 (bps, 1bp = 0.01%)
RL_TRANSACTION_COST_BPS = 10.0
# yfinance로 내려받을 과거 가격 데이터 기간 (년). 이 중 RL_TRAIN_TEST_SPLIT
# 비율만큼 학습에 쓰고 나머지 최근 구간은 백테스트/드리프트 점검용으로 보류.
RL_TRAIN_YEARS = 6
RL_TRAIN_TEST_SPLIT = 0.8
# 학습 에피소드 길이 (거래일). 학습 구간 내 랜덤 시작일에서 이 길이만큼 진행.
RL_EPISODE_LENGTH_DAYS = 126
RL_MODEL_PATH = "rl/models/ppo_portfolio.zip"
# 리밸런싱 시 이 금액(달러) 미만의 목표-현재 비중 차이는 무시한다 (잦은 소액 거래 방지)
RL_REBALANCE_MIN_TRADE_USD = 5.0
# daily_run.py 드리프트 점검: 보류 구간 백테스트 최대낙폭(MDD)이 이 값을
# 넘으면 이상 상황으로 보고 리밸런싱 없이 종료한다.
RL_DRIFT_MAX_DRAWDOWN = 0.35
