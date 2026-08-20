# 매크로 블로그 기반 RL 포트폴리오 에이전트

S&P 500 전 종목에 비중을 배분하는 PPO(stable-baselines3) 강화학습
포트폴리오 에이전트입니다. 가격 기반 기술적 지표에 더해, "헨리의
퀀트대학"(blog.naver.com/leebisu) 블로그가 매일 올리는 "글로벌 매크로
트렌드" 글을 크롤링해 KR-FinBert-SC로 감성 점수화한 매크로 시그널을
관측(observation)에 함께 반영합니다.

과거에는 뉴스(Alpha Vantage)+애널리스트 컨센서스(Finnhub) 기반 규칙형
매매 전략(`strategy.py`)이 별도로 있었으나, 매크로 블로그 기반 RL로
대체되어 제거되었습니다. 이 저장소의 매매 로직은 `rl/` 하나뿐입니다.

## 폴더 구조

```
.
├── config.py             # 설정값 (토스 API, RL 하이퍼파라미터)
├── data_pipeline.py      # S&P500/NASDAQ100 종목 유니버스 조회 (RL 학습 유니버스용)
├── broker.py             # 토스증권 Open API 클라이언트 (계좌/주문, 기본 드라이런)
├── rl/
│   ├── price_data.py           # yfinance 과거 가격 조회 + 기술적 지표 계산
│   ├── macro_blog.py           # 매크로 블로그 크롤링 + KR-FinBert-SC 감성 점수화
│   ├── macro_daily_scores.json # 날짜별 매크로 점수 저장소 (git 추적)
│   ├── trading_env.py          # PortfolioEnv (gymnasium 커스텀 환경, 매크로 시그널 포함)
│   ├── train.py                # PPO 학습 (S&P500 유니버스 + 매크로 시그널)
│   ├── backtest.py              # 보류 구간 백테스트 + buy&hold/현금 베이스라인 비교
│   ├── rl_strategy.py           # 학습된 정책으로 실거래 리밸런싱
│   ├── paper_trading.py         # 드라이런 중 대시보드용 가상 포트폴리오 가치 시뮬레이션
│   ├── daily_run.py             # 매 거래일 자동 드리프트 점검 + 리밸런싱 진입점
│   ├── daily_log.py             # 실행 기록 누적(daily_run_history.json) + README 자동 갱신
│   ├── daily_run_history.json   # 일별 실행 기록 누적 데이터 (git 추적)
│   └── models/                   # 학습된 모델(.zip)과 메타데이터(.meta.json) 저장 위치
├── dashboard/
│   └── index.html        # Vercel 실시간 대시보드 (daily_run_history.json 시각화)
├── tests/
├── .cache/                # 가격 데이터/유니버스/매크로 블로그 본문 캐시 (git 제외)
└── output/                # rl_backtest_*.csv/png 생성 위치
```

## 설치

```bash
pip install -r requirements.txt   # stable-baselines3, gymnasium, yfinance, transformers(+torch) 포함, 용량 큼
cp .env.example .env
# .env 파일을 열어 토스증권 client_id/secret을 입력
```

## 매크로 블로그 시그널 (rl/macro_blog.py)

블로그의 "Global Macro" 카테고리 글을 불릿 단위로 쪼개 KR-FinBert-SC로
각각 감성 점수화(P(positive) - P(negative))한 뒤 평균해, 하루 1개
스칼라 점수로 `rl/macro_daily_scores.json`에 누적합니다.

```bash
python -m rl.macro_blog             # 오늘자 글 점수화 + 저장
python -m rl.macro_blog --push      # 저장 후 git commit/push까지
python -m rl.macro_blog --backfill  # 카테고리 전체 과거 글을 채운다 (최초 1회용)
```

**한계**: 이 블로그의 "Global Macro" 카테고리는 2025-11-09부터 시작했습니다.
그래서 RL 학습 구간도 6년치 가격 데이터 전체가 아니라 매크로 데이터가 있는
시점부터로 맞춥니다 (`rl/train.py`가 `macro_blog.get_macro_series()`로
자동 판단) - 학습에 쓸 수 있는 거래일이 약 150~200일로 짧아, 종목 수(약
480개) 대비 데이터가 부족해 과적합 위험이 있습니다. 향후 블로그 히스토리가
쌓일수록 개선됩니다.

### 매일 자동 크롤링 (10시, launchd)

`~/Library/LaunchAgents/com.minsung.rl-macro-blog.plist`가 매일 오전
10시(블로그가 보통 그 전에 당일 글을 올림)에 `python -m rl.macro_blog --push`를
실행해 그날 글을 점수화하고 저장소에 push합니다. `rl.daily_run`(아래,
30분 간격)은 이 저장소를 읽기만 하고 크롤링은 하지 않습니다 - 크롤링/점수화는
이 잡이 전담합니다.

## 학습 (rl/train.py)

```bash
python -m rl.train                    # 기본 200,000 timesteps
python -m rl.train --timesteps 20000  # 빠른 스모크 테스트용
```

`data_pipeline.get_sp500_tickers()`로 S&P 500 종목 유니버스를 가져오고
(6년치 데이터가 없는 최근 상장 종목은 `rl.price_data.load_dataset_filtered()`가
제외), `rl.macro_blog.get_macro_series()`로 매크로 시그널이 존재하는
시점부터 학습 구간을 잡습니다. `config.RL_TRAIN_TEST_SPLIT`(기본 80%)
지점까지만 학습하고, 나머지는 보류(holdout) 구간으로 남겨 백테스트에
사용합니다. 결과는 `rl/models/ppo_portfolio.zip`(모델)과 `.meta.json`
(종목/윈도/분할 날짜/`macro_enabled` 등 메타데이터)으로 저장됩니다.

## 백테스트

```bash
python -m rl.backtest
```

보류 구간에서 학습된 정책을 결정적으로(deterministic) 실행해 동일종목
동일가중 buy&hold, 전량 현금 보유와 CAGR/연변동성/Sharpe/최대낙폭(MDD)을
비교합니다. `output/rl_backtest_<날짜>.csv`(자산곡선 데이터)와 `.png`(그래프)를
저장합니다.

## 토스증권 Open API 연동 (실거래, broker.py)

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

두 조건을 모두 충족해야 실제 주문이 나갑니다. `rl_strategy.py`는 이
안전장치를 그대로 사용하며 실거래 여부를 직접 판단하지 않습니다 -
`TOSS_LIVE_TRADING=true` 전환 자체가 유일한 게이트입니다.

이 계좌는 미국 주식 전용으로 운용합니다. `broker.create_order()`가 KRX
종목코드(6자리 숫자)를 코드 레벨에서 차단합니다.

## 실거래 자동화 (daily_run.py)

```bash
python -m rl.daily_run
```

매 거래일: 최신 가격으로 ① 보류 구간 백테스트를 다시 돌려 MDD가
`config.RL_DRIFT_MAX_DRAWDOWN`(기본 35%)을 넘는지 점검(드리프트 감지 시
주문 없이 종료) → ② 통과하면 `rl_strategy.py`로 오늘자 목표 비중을 계산해
`broker.create_order()`로 리밸런싱 주문을 냅니다. **자동 재학습은 하지
않습니다** — 모델이 오래됐다고 판단되면 `python -m rl.train`을 수동
재실행하세요.

드라이런(`TOSS_LIVE_TRADING=false`) 동안은 실계좌가 절대 바뀌지 않으므로,
기록/대시보드용 포트폴리오 가치는 `rl/paper_trading.py`가 직전 기록의
가치·비중 + 오늘 종가로(학습 때와 같은 갱신식) 시뮬레이션합니다. 시작 자본은
최초 실행 시 실계좌 buying power를 그대로 가져옵니다.

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

매크로 블로그 크롤링은 별도로 매일 10시에 도는 `com.minsung.rl-macro-blog.plist`가
전담합니다 (위 "매크로 블로그 시그널" 절 참고).

리눅스 서버(한국 리전)에서 돌린다면 cron도 괜찮지만, 서버는 보통 항상 켜져
있어 "잠들었다 깨어남" 문제 자체가 없기 때문입니다:

```
0 8 * * 1-5 cd /path/to/repo && /path/to/repo/.venv/bin/python -m rl.daily_run --push >> rl_daily_run.log 2>&1
```

## 실시간 대시보드 (Vercel)

**https://minsung-investment-dashboard.vercel.app**

`dashboard/index.html` 정적 페이지가 이 저장소의 `rl/daily_run_history.json`
(raw GitHub)을 60초마다 읽어 포트폴리오 가치 추이, 일간 수익률, 종목별 목표
비중, 전체 기록 표를 보여줍니다. 별도 서버 없이 저장소에 기록이 push되면
자동 반영됩니다(GitHub raw 캐시로 수 분 지연 가능). 재배포:

```bash
cd dashboard && npx vercel deploy --prod
```

## 일별 실행 기록 (자동 생성)

`rl/daily_run.py`가 완주할 때마다(드리프트 통과 + 리밸런싱 계산 완료) 그날의
포트폴리오 가치·전일 대비 수익률·종목별 목표 비중을 `rl/daily_run_history.json`에
누적 기록하고, 아래 표/그래프 구간을 자동으로 재생성합니다(수동 편집 불필요 —
다음 실행 때 덮어써집니다).

<!-- RL_DAILY_LOG_START -->
실행할 때마다 이 표/그래프가 자동으로 갱신됩니다 (`rl/daily_run.py`가
`rl/daily_run_history.json`에 결과를 추가하고 이 구간을 재생성합니다 -
수동으로 이 마커(`RL_DAILY_LOG_START`/`_END`) 사이를 직접 편집하지
마세요, 다음 실행 때 덮어써집니다). 드리프트 임계값 초과로 리밸런싱을
건너뛴 실행은 기록하지 않습니다.

| 날짜 | 포트폴리오 가치 | 일간 수익률 | AAPL | AMZN | GOOGL | MSFT | NVDA | 현금 | 비고 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-21 | $72.44 | - | 17.0% | 16.6% | 15.9% | 17.5% | 16.1% | 16.9% | (dry-run) |
<!-- RL_DAILY_LOG_END -->

## 향후 확장

- 매크로 블로그 히스토리가 쌓일수록(현재 ~9개월) 학습 구간을 늘려 과적합
  위험을 낮출 수 있음
- 다중 자산군 확장(현재는 S&P 500 + 현금, 채권/원자재 등 다른 자산군은 미포함)
- 거래비용/슬리피지 가정을 더 정교하게 반영
- 매크로 블로그 크롤러(`rl/macro_blog.py`)는 네이버 블로그 내부 위젯
  API(공식 문서 없음)에 의존 - 블로그 구조가 바뀌면 깨질 수 있음

## 참고 (중요)

- 이 신호는 리서치/교육 목적이며 투자 자문이 아닙니다.
- `broker.py`로 실제 주문을 실행하는 것은 전적으로 본인 책임이며,
  `TOSS_LIVE_TRADING=true` 전환은 신중하게 결정해야 합니다.
