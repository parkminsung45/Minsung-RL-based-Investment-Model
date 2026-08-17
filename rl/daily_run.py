"""
매 거래일 자동 실행 진입점: 최신 가격으로 보류(holdout) 구간 백테스트를 다시
돌려 드리프트(성과 급락)를 점검하고, 통과하면 오늘자 리밸런싱 주문을 낸다.
완주한 실행(드리프트 통과 + 리밸런싱 계산 완료)은 rl/daily_log.py를 통해
rl/daily_run_history.json에 기록되고 README.md의 일별 실행 기록 섹션이
자동 갱신된다.

자동 재학습은 하지 않는다 - 모델 재학습은 README에 안내된 대로 `python -m
rl.train`을 월 1회 등 별도 주기로 수동 실행할 것. 드리프트가 감지되면 주문
없이 로그만 남기고 종료한다(히스토리에도 기록하지 않음 - 부분 실행 데이터로
추이 그래프를 흐리지 않기 위해서).

config.TOSS_LIVE_TRADING=false(기본값)인 동안은 broker.create_order()가
실제 주문을 보내지 않고 dry-run으로만 동작한다 (기존 안전장치 그대로 적용).

실행 방법: python -m rl.daily_run
    --push 옵션을 주면 실행 후 갱신된 기록(rl/daily_run_history.json, README.md)을
    git commit/push한다 - Vercel 대시보드가 GitHub raw JSON을 읽으므로, push까지
    해야 대시보드에 당일 기록이 반영된다.

토스증권 API가 해외(클라우드) IP를 차단해(2026-08-17 확인) 반드시 한국 IP의
컴퓨터에서 실행해야 한다. 정확한 시각의 cron 대신, 이 컴퓨터가 켜져 있고
인터넷에 연결된 동안 launchd가 일정 주기(예: 30분)로 계속 이 스크립트를
호출하도록 등록하는 걸 권장한다(README "실거래 자동화" 참고) - 잠자는 동안
놓친 실행을 cron은 그냥 건너뛰지만, launchd는 깨어나는 대로 다시 시도한다.
그래서 run()은 자체적으로 (1) 주말이면 건너뛰고 (2) 오늘 이미 완주한 실행
기록이 있으면 건너뛰어, 하루에 여러 번 트리거돼도 안전(idempotent)하다.
"""
import argparse
import os
import subprocess
from datetime import date
from typing import Optional

import broker
import config
from rl import backtest, daily_log, rl_strategy

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _already_ran_today(today_iso: str) -> bool:
    history = daily_log.load_history()
    return bool(history) and history[-1]["date"] == today_iso


def check_drift(model, meta) -> bool:
    """
    보류 구간 백테스트의 최대낙폭(MDD)이 config.RL_DRIFT_MAX_DRAWDOWN을 넘으면
    이상 상황으로 보고 False를 반환한다 (이번 실행의 리밸런싱을 생략).
    """
    feature_df, close_df = backtest.load_holdout(meta)
    result = backtest.run_backtest(
        model, feature_df, close_df, meta["tickers"], meta["window"], meta["transaction_cost_bps"]
    )
    mdd = abs(result["rl"]["metrics"]["max_drawdown"])
    print(f"[drift-check] 보류 구간 MDD={mdd:.2%} (임계값={config.RL_DRIFT_MAX_DRAWDOWN:.0%})")
    return mdd <= config.RL_DRIFT_MAX_DRAWDOWN


def push_history() -> None:
    """
    갱신된 기록 파일과 README만 커밋해 push한다. 다른 변경사항은 건드리지
    않도록 파일을 명시하고, 실패해도 리밸런싱 자체는 이미 끝났으므로 경고만
    남긴다 (푸시가 안 되면 대시보드에 당일 기록만 늦게 반영될 뿐이다).
    """
    files = ["rl/daily_run_history.json", "README.md"]
    try:
        subprocess.run(["git", "add", *files], cwd=_REPO_ROOT, check=True)
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", *files], cwd=_REPO_ROOT
        )
        if diff.returncode == 0:
            print("[push] 변경 사항 없음 - 커밋/푸시 생략")
            return
        subprocess.run(
            ["git", "commit", "-m", f"일별 실행 기록: {date.today().isoformat()}", "--", *files],
            cwd=_REPO_ROOT, check=True,
        )
        subprocess.run(["git", "push"], cwd=_REPO_ROOT, check=True)
        print("[push] 기록 커밋/푸시 완료 - 대시보드에 곧 반영됩니다")
    except subprocess.CalledProcessError as exc:
        print(f"[push] 경고: git 커밋/푸시 실패 ({exc}) - 기록은 로컬에 저장돼 있으니 수동으로 push하세요")


def run(push: bool = False, today: Optional[date] = None):
    today = today or date.today()

    if today.weekday() >= 5:  # 5=토, 6=일 - 미국 증시 휴장
        print(f"[skip] {today.isoformat()}은 주말입니다 - 리밸런싱 생략")
        return []

    if _already_ran_today(today.isoformat()):
        print(f"[skip] 오늘({today.isoformat()}) 이미 완주한 실행 기록이 있습니다 - 생략")
        return []

    print("[1/3] 모델 로드 및 드리프트 점검 중...")
    model, meta = rl_strategy.load_model_and_meta()

    if not check_drift(model, meta):
        print("드리프트 임계값 초과 - 이번 실행에서는 리밸런싱을 건너뜁니다.")
        return []

    if not config.TOSS_CLIENT_ID or not config.TOSS_CLIENT_SECRET:
        raise RuntimeError(".env 파일에 TOSS_CLIENT_ID와 TOSS_CLIENT_SECRET을 설정하세요.")

    print("[2/3] 계좌 조회 중...")
    client = broker.TossClient(config.TOSS_CLIENT_ID, config.TOSS_CLIENT_SECRET)
    # 실계좌 검증 완료(2026-08-17): 계좌 목록은 result 키 아래 리스트로 온다.
    account_list = broker.get_accounts(client).get("result") or []
    if not account_list:
        raise RuntimeError("조회된 계좌가 없습니다.")
    client.set_account(account_list[0]["accountSeq"])

    print("[3/3] 리밸런싱 주문 계산/실행 중...")
    results, snapshot = rl_strategy.run(client, model=model, meta=meta)
    for r in results:
        print(r)

    actions = {r["symbol"]: r["action"] for r in results}
    daily_log.append_entry(
        portfolio_value=snapshot["portfolio_value"],
        weights=snapshot["weights"],
        actions=actions,
        dry_run=not config.TOSS_LIVE_TRADING,
    )
    print(f"기록 완료: {daily_log.HISTORY_PATH}, README.md 일별 실행 기록 섹션 갱신")

    if push:
        push_history()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--push", action="store_true",
        help="실행 후 갱신된 기록/README를 git commit & push (대시보드 반영)",
    )
    args = parser.parse_args()
    run(push=args.push)
