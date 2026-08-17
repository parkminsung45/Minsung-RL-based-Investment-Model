# 뉴스 + 애널리스트 컨센서스 감성분석 파이프라인

미국 주식 종목별로 뉴스 감성점수와 애널리스트 컨센서스(추천등급/목표주가)를
결합해 하나의 시그널(-1 ~ 1)로 만드는 파이프라인입니다.

## 왜 "애널리스트 리포트 원문"이 아닌가?

실제 애널리스트 리포트 PDF(예: 골드만삭스, 모건스탠리 리포트)는 대부분
기관 전용 유료 데이터라 공개 API로는 가져올 수 없습니다. 대신 여러 애널리스트
의견을 요약한 **공개 컨센서스 데이터**(매수/보유/매도 추천 분포, 평균 목표주가)를
사용합니다. 이는 "여러 리포트 결론의 요약본"으로 볼 수 있습니다.

## 1. 준비물 (본인이 직접 가입)

| 서비스 | 용도 | 가입 링크 |
|---|---|---|
| Alpha Vantage | 뉴스 + 감성점수 | https://www.alphavantage.co/support/#api-key |
| Finnhub | 애널리스트 컨센서스 | https://finnhub.io/register |

두 곳 모두 무료 티어로 시작 가능합니다 (호출 횟수 제한 있음).

## 2. 설치

```bash
pip install -r requirements.txt
cp .env.example .env
# .env 파일을 열어 발급받은 키를 입력
```

## 폴더 구조

```
.
├── main.py              # 관심종목(WATCHLIST) 뉴스+애널리스트 결합 시그널 실행
├── scan_universe.py     # S&P500+NASDAQ100 전체 종목 애널리스트 스코어 스캔
├── config.py            # 설정값 (티커, 가중치, API 키 로딩)
├── data_pipeline.py     # 뉴스/애널리스트/유니버스/재무건전성 수집 + 시그널 결합 + FinBERT
├── broker.py            # 토스증권 Open API 클라이언트 (계좌/주문, 기본 드라이런)
├── strategy.py          # 점수 -> 매수/매도 판단, 포지션 사이징, 재무 건전성 필터
├── run_strategy.py      # strategy.py를 실제 계좌에 대해 실행하는 진입점
├── rl/                   # 강화학습 기반 포트폴리오 에이전트 (실험적, 9번 참고)
│   ├── price_data.py    # yfinance 과거 가격 조회 + 기술적 지표 계산
│   ├── trading_env.py   # PortfolioEnv (gymnasium 커스텀 환경)
│   ├── train.py         # PPO 학습
│   ├── backtest.py      # 보류 구간 백테스트 + buy&hold/현금 베이스라인 비교
│   ├── rl_strategy.py   # 학습된 정책으로 실거래 리밸런싱 (strategy.py와 별개 경로)
│   ├── daily_run.py     # 매 거래일 자동 드리프트 점검 + 리밸런싱 진입점
│   ├── daily_log.py     # 실행 기록 누적(daily_run_history.json) + README 자동 갱신
│   ├── daily_run_history.json  # 일별 실행 기록 누적 데이터 (git 추적)
│   └── models/           # 학습된 모델(.zip)과 메타데이터(.meta.json) 저장 위치
├── dashboard/
│   └── index.html        # Vercel 실시간 대시보드 (daily_run_history.json 시각화)
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_broker.py
│   ├── test_strategy.py
│   ├── test_price_data.py
│   ├── test_trading_env.py
│   └── test_daily_log.py
├── .cache/               # data_pipeline.py/rl 가격 데이터 캐시 (git 제외)
└── output/               # signals_*.csv, universe_analyst_scores_*.csv, rl_backtest_*.csv/png 생성 위치
```

## 3. 실행

```bash
python main.py
```

`output/signals_YYYY-MM-DD.csv` 파일이 생성됩니다. 컬럼:

- `ticker`: 종목 코드
- `news_score`: 뉴스 감성 점수 (-1 ~ 1, relevance로 가중평균)
- `analyst_score`: 애널리스트 컨센서스 점수 (-1 ~ 1)
- `combined_score`: 최종 결합 시그널
- `target_mean_price`: 애널리스트 평균 목표주가
- `num_articles`: 수집된 기사 수

## 4. 설정 변경 (config.py)

- `WATCHLIST`: 뉴스+애널리스트 결합 시그널(main.py)을 계산할 소수 관심종목.
  Alpha Vantage 무료 티어(하루 25회)로 감당 가능한 규모로 유지할 것
- `NEWS_WEIGHT`, `ANALYST_WEIGHT`: 뉴스 vs 애널리스트 가중치 (합=1)
- `NEWS_FETCH_DELAY_SEC`: Alpha Vantage 무료 티어 호출 제한 대응 대기시간
- `FINNHUB_UNIVERSE_DELAY_SEC`: scan_universe.py에서 Finnhub 호출 간 대기시간

## 4-1. S&P500 + NASDAQ100 전체 스캔 (scan_universe.py)

```bash
python scan_universe.py
```

`data_pipeline.py`의 유니버스 조회 함수가 stockanalysis.com(실패 시 위키피디아, 그마저
실패하면 `.cache/universe_cache.json`의 직전 결과)에서 S&P500·NASDAQ100
구성종목(중복 제거 후 약 520개)을 매번 새로 가져와 Finnhub 애널리스트
컨센서스 점수만 계산합니다.

뉴스 감성(Alpha Vantage)은 이 규모에서 하루 호출 한도(25회)를 크게 초과하므로
제외했습니다. 뉴스까지 포함한 결합 시그널이 필요한 종목은 `WATCHLIST`에
추가해 `main.py`로 계산하세요.

`output/universe_analyst_scores_YYYY-MM-DD.csv` 컬럼:

- `ticker`, `analyst_score`, `strong_buy`, `buy`, `hold`, `sell`, `strong_sell`

전체 스캔은 티커당 `FINNHUB_UNIVERSE_DELAY_SEC`(기본 1초)씩 대기하므로
약 8~10분 소요됩니다.

## 5. FinBERT로 자체 감성분석 추가 (선택)

`data_pipeline.py`의 `score_texts()`를 사용하면 Alpha Vantage 자체
점수와 별개로, 금융특화 언어모델(FinBERT)로 뉴스 제목을 직접 채점해
두 점수를 앙상블할 수 있습니다. 최초 실행 시 모델(~400MB)을 다운로드합니다.

```python
from data_pipeline import score_texts
titles = [a["title"] for a in news_by_ticker["AAPL"]]
finbert_scores = score_texts(titles)
```

## 6. 매일 자동 실행 (선택)

리눅스/맥에서 cron으로 매일 아침 실행 예시:

```
0 8 * * 1-5 cd /path/to/news_analyst_pipeline && /usr/bin/python3 main.py >> log.txt 2>&1
```

## 7. 토스증권 Open API 연동 (실거래, broker.py)

`broker.py`는 [토스증권 Open API](https://developers.tossinvest.com/docs)로
계좌 조회와 실제 주문 실행을 담당합니다. client_id/client_secret은 토스증권
WTS 로그인 후 설정 > Open API에서 발급받아 `.env`에 입력합니다
(`TOSS_CLIENT_ID`, `TOSS_CLIENT_SECRET`).

**⚠️ 이 API는 샌드박스/모의투자 환경이 없습니다.** 주문 생성 요청은 즉시
실제 계좌·실제 자금에 반영됩니다. 그래서 `broker.py`는 이중 안전장치로
동작합니다.

1. `config.TOSS_LIVE_TRADING`이 `false`(기본값)인 동안은 실제로 주문을
   보내지 않고, 보낼 요청 내용만 출력/반환합니다 (dry-run).
2. `.env`에서 `TOSS_LIVE_TRADING=true`로 실거래를 켠 상태에서도, 각 함수
   호출 시 `confirm=True`를 명시하지 않으면 실행을 거부합니다.

두 조건을 모두 충족해야 실제 주문이 나갑니다.

```python
import broker
import config

client = broker.TossClient(config.TOSS_CLIENT_ID, config.TOSS_CLIENT_SECRET)
accts = broker.get_accounts(client)          # 계좌 목록 조회 (읽기 전용)
# client.set_account(accountSeq) 로 이후 요청에 쓸 계좌를 지정

result = broker.create_order(
    client, symbol="AAPL", side="BUY", order_type="MARKET", quantity=1,
)
# TOSS_LIVE_TRADING=false 인 동안은 항상 {"dry_run": True, "would_send": {...}} 반환
```

현재 API 신청은 승인 대기 중이라 계좌/주문 엔드포인트는 실제 자격증명으로
아직 검증하지 못했습니다. 승인되면 `get_accounts()`부터 먼저 호출해
응답 형식을 확인한 뒤 필요하면 파싱 로직을 다듬어야 합니다.

## 8. 매매 전략 (strategy.py, run_strategy.py)

`signals_*.csv`(WATCHLIST, combined_score)와 `universe_analyst_scores_*.csv`
(그 외 종목, analyst_score)의 점수를 실제 매수/매도 판단으로 연결합니다.

규칙:

1. `score > config.BUY_THRESHOLD`(기본 0.3) -> 매수 후보
   - 이미 보유 중이면 건너뜀
   - **재무 건전성 필터**: Finnhub Basic Financials(`stock/metric`)로
     순이익률(TTM) > 0, ROE(TTM) > 0, 부채비율(D/E) < 2.0을 확인해
     통과하지 못하면 매수하지 않음 (`config.MIN_NET_MARGIN`,
     `config.MIN_ROE`, `config.MAX_DEBT_TO_EQUITY`)
   - 매수가능금액(buying power)의 `config.POSITION_SIZE_PCT`(기본 5%)만큼
     시장가 매수
2. `score < config.SELL_THRESHOLD`(기본 -0.3) -> 보유 중이면 전량 시장가 매도
3. 그 외 -> 홀드

```bash
python run_strategy.py
```

`broker.create_order()`의 안전장치(기본 드라이런, `TOSS_LIVE_TRADING=true`+
`confirm=True` 필요)가 그대로 적용됩니다. `strategy.run()`은 `confirm`을
`config.TOSS_LIVE_TRADING`과 동일한 값으로 자동 전달하므로, 사람이 매번
확인할 필요 없이 "실거래를 켠다"는 결정 하나가 유일한 게이트가 됩니다.

**⚠️ 임계값 보정 필요**: Finnhub 애널리스트 점수는 매도 의견이 구조적으로
드물어 양수 쪽에 쏠려 있습니다. 519개 유니버스 스캔 기준 중앙값이 0.41이라,
기본 임계값(0.3)으로는 약 74%(385개)가 매수 후보로 잡힙니다. 지금은 로직
구현을 우선하고 임계값 자체는 나중에 다시 조정하기로 했습니다 — 실제
운용 전에 `BUY_THRESHOLD`/`SELL_THRESHOLD`를 상위 N% 방식 등으로
재검토할 것.

`accounts`, `holdings`, `buying-power`, `sellable-quantity` 응답의 정확한
필드명은 토스증권 API 승인 전이라 실제 자격증명으로 검증하지 못했습니다.
`strategy.py`의 `_extract_value()`가 후보 키 여러 개를 시도하도록 만들어
뒀지만, 승인 후 실제 응답을 보고 정리해야 합니다.

## 9. 강화학습 기반 포트폴리오 에이전트 (실험적, rl/)

`strategy.py`의 종목별 고정 임계값 규칙과는 별개로, `WATCHLIST` 종목들에
비중을 동시에 배분하는 PPO(stable-baselines3) 기반 포트폴리오 에이전트를
`rl/`에 추가했습니다. `strategy.py`/`run_strategy.py`는 그대로 남아있고
(수정하지 않음), 둘 중 하나를 선택해서 실행하면 됩니다.

**중요한 한계**: 뉴스(Alpha Vantage)·애널리스트 컨센서스(Finnhub)는 무료
API로 수년치 일별 히스토리를 구할 수 없습니다. 그래서 이 RL 에이전트의
state는 **가격 기반 기술적 지표만** 사용합니다(일간수익률, 5/20/60일
이동평균수익률, 20일 변동성, RSI(14), 거래량 z-score — `rl/price_data.py`).
`main.py`의 뉴스+애널리스트 `combined_score`는 이 파이프라인에 아직
반영되지 않습니다.

### 설치

```bash
pip install -r requirements.txt   # stable-baselines3, gymnasium, yfinance, matplotlib(+torch) 포함, 용량 큼
```

### 학습

```bash
python -m rl.train                    # 기본 200,000 timesteps
python -m rl.train --timesteps 20000  # 빠른 스모크 테스트용
```

`config.RL_TRAIN_YEARS`(기본 6년)만큼 yfinance에서 일별 가격을 내려받아
`config.RL_TRAIN_TEST_SPLIT`(기본 80%) 지점까지만 학습하고, 최근 20%는
보류(holdout) 구간으로 남겨 미래 데이터 누수 없이 백테스트에 사용합니다.
결과는 `rl/models/ppo_portfolio.zip`(모델)과 `.meta.json`(종목/윈도/분할
날짜 등 메타데이터)으로 저장됩니다.

### 백테스트

```bash
python -m rl.backtest
```

보류 구간에서 학습된 정책을 결정적으로(deterministic) 실행해 동일종목
동일가중 buy&hold, 전량 현금 보유와 CAGR/연변동성/Sharpe/최대낙폭(MDD)을
비교합니다. `output/rl_backtest_<날짜>.csv`(자산곡선 데이터)와 `.png`(그래프)를
저장합니다.

### 실거래 자동화 (daily_run.py)

```bash
python -m rl.daily_run
```

매 거래일: 최신 가격으로 ① 보류 구간 백테스트를 다시 돌려 MDD가
`config.RL_DRIFT_MAX_DRAWDOWN`(기본 35%)을 넘는지 점검(드리프트 감지 시
주문 없이 종료) → ② 통과하면 `rl_strategy.py`로 오늘자 목표 비중을 계산해
`broker.create_order()`로 리밸런싱 주문을 냅니다. **자동 재학습은 하지
않습니다** — 모델이 오래됐다고 판단되면 `python -m rl.train`을 월 1회
등 별도 주기로 수동 재실행하세요.

`broker.create_order()`의 기존 이중 안전장치(`TOSS_LIVE_TRADING=false`
기본 드라이런, `confirm=True` 필요)가 `strategy.run()`과 동일하게
그대로 적용됩니다 — `rl_strategy.py`도 실거래 여부를 직접 판단하지 않습니다.

**⚠️ 반드시 한국 IP에서 실행해야 합니다.** 토스증권 API는 해외(클라우드) IP의
OAuth 토큰 발급을 403으로 거부합니다(2026-08-17 GitHub Actions에서 실제 확인 -
로컬/한국 IP에서는 동일 코드·키로 정상 동작). 그래서 GitHub Actions 등
해외 리전 클라우드로는 자동화할 수 없고, **한국에 있는 컴퓨터**에서 돌려야
합니다. `.github/workflows/daily_run.yml`은 참고용으로 남겨뒀지만 스케줄은
꺼두었습니다(수동 실행만 가능) — 이 안에서 실행하면 항상 실패합니다.

**cron 대신 launchd(macOS)를 권장합니다.** 정해진 시각에만 도는 cron은 그
시각에 컴퓨터가 잠들어 있으면 그날 실행을 그냥 건너뜁니다. `run()`은 자체적으로
(1) 주말이면 스킵, (2) 오늘 이미 완주한 실행 기록이 있으면 스킵하도록 만들어져
있어 하루에 여러 번 호출해도 안전(idempotent)하므로, launchd로 컴퓨터가 켜져
있는 동안 짧은 주기(예: 30분)로 계속 시도하게 등록하면 됩니다 - 잠들었다
깨어나도 다음 주기에 자동으로 따라잡습니다. `~/Library/LaunchAgents/`에
아래처럼 등록(`--push`로 실행 후 기록을 자동 커밋/푸시해 대시보드에 반영):

```xml
<!-- ~/Library/LaunchAgents/com.minsung.rl-daily-run.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
    <key>Label</key><string>com.minsung.rl-daily-run</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/repo/.venv/bin/python</string>
        <string>-m</string><string>rl.daily_run</string><string>--push</string>
    </array>
    <key>WorkingDirectory</key><string>/path/to/repo</string>
    <key>StartInterval</key><integer>1800</integer>
    <key>RunAtLoad</key><true/>
    <key>StandardOutPath</key><string>/path/to/repo/rl_daily_run.log</string>
    <key>StandardErrorPath</key><string>/path/to/repo/rl_daily_run.log</string>
</dict></plist>
```

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.minsung.rl-daily-run.plist
```

리눅스 서버(한국 리전)에서 돌린다면 cron도 괜찮지만, 서버는 보통 항상 켜져
있어 "잠들었다 깨어남" 문제 자체가 없기 때문입니다:

```
0 8 * * 1-5 cd /path/to/repo && /path/to/repo/.venv/bin/python -m rl.daily_run --push >> rl_daily_run.log 2>&1
```

### 실시간 대시보드 (Vercel)

**https://minsung-investment-dashboard.vercel.app**

`dashboard/index.html` 정적 페이지가 이 저장소의 `rl/daily_run_history.json`
(raw GitHub)을 60초마다 읽어 포트폴리오 가치 추이, 일간 수익률, 종목별 목표
비중, 전체 기록 표를 보여줍니다. 별도 서버 없이 저장소에 기록이 push되면
자동 반영됩니다(GitHub raw 캐시로 수 분 지연 가능). 재배포:

```bash
cd dashboard && npx vercel deploy --prod
```

### 일별 실행 기록 (자동 생성)

`rl/daily_run.py`가 완주할 때마다(드리프트 통과 + 리밸런싱 계산 완료) 그날의
포트폴리오 가치·전일 대비 수익률·종목별 목표 비중을 `rl/daily_run_history.json`에
누적 기록하고, 아래 표/그래프 구간을 자동으로 재생성합니다(수동 편집 불필요 —
다음 실행 때 덮어써집니다). 그라운딩 회귀 스크립트의 히스토리 기록 패턴과
동일한 방식입니다.

<!-- RL_DAILY_LOG_START -->
실행할 때마다 이 표/그래프가 자동으로 갱신됩니다 (`rl/daily_run.py`가
`rl/daily_run_history.json`에 결과를 추가하고 이 구간을 재생성합니다 -
수동으로 이 마커(`RL_DAILY_LOG_START`/`_END`) 사이를 직접 편집하지
마세요, 다음 실행 때 덮어써집니다). 드리프트 임계값 초과로 리밸런싱을
건너뛴 실행은 기록하지 않습니다.

| 날짜 | 포트폴리오 가치 | 일간 수익률 | AAPL | AMZN | GOOGL | MSFT | NVDA | 현금 | 비고 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-17 | $72.56 | - | 16.8% | 16.9% | 16.7% | 16.9% | 16.0% | 16.7% | (dry-run) |
<!-- RL_DAILY_LOG_END -->

### 향후 확장

- LLM 기반 뉴스/공시 리서치 신호를 RL state에 추가 피처로 결합 (현재는
  가격 기술적 지표만 사용). 참고: [LinqAlpha 리더보드](https://linqalpha.com/leaderboard)는
  금융 판단에서 LLM의 섹터/시가총액/모멘텀 편향성을 측정하는 벤치마크로,
  도입 시 편향 낮은 모델을 우선 검토할 것.
- 다중 자산군 확장(현재는 `WATCHLIST` 5종목 + 현금)
- 거래비용/슬리피지 가정을 더 정교하게 반영

## 다음 단계

- `BUY_THRESHOLD`/`SELL_THRESHOLD` 재보정 (상위 N% 방식 등)
- 토스증권 API 승인 후 `broker.py`/`strategy.py`/`rl_strategy.py` 실제 계좌로 검증
- RL 에이전트 하이퍼파라미터(윈도, 에피소드 길이, 거래비용 가정) 튜닝

## 참고 (중요)

- 이 신호는 리서치/교육 목적이며 투자 자문이 아닙니다.
- `broker.py`로 실제 주문을 실행하는 것은 전적으로 본인 책임이며,
  `TOSS_LIVE_TRADING=true` 전환은 신중하게 결정해야 합니다.
