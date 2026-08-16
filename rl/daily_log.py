"""
daily_run.py가 완주할 때마다(드리프트 점검 통과 + 리밸런싱 계산 완료) 결과를
rl/daily_run_history.json에 누적 기록하고, README.md의
RL_DAILY_LOG_START/END 마커 사이를 표 + mermaid xychart-beta 그래프로
재생성한다. scripts/grounding_regression.py의 "실행할 때마다 히스토리에
추가하고 README를 자동 갱신" 패턴을 그대로 따른다 - 마커 사이를 수동으로
편집하지 말 것 (다음 실행 때 덮어써진다).

드리프트 임계값 초과로 리밸런싱을 건너뛴 실행은 기록하지 않는다 - 포트폴리오
가치/비중 스냅샷 자체가 없고, 부분 실행 데이터로 추이 그래프를 흐리지 않기
위해서다.
"""
import json
import os
from datetime import date as date_cls
from typing import Dict, List, Optional

_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(_DIR, "daily_run_history.json")
README_PATH = os.path.join(_DIR, "..", "README.md")
README_LOG_START = "<!-- RL_DAILY_LOG_START -->"
README_LOG_END = "<!-- RL_DAILY_LOG_END -->"


def load_history() -> List[Dict]:
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_history(history: List[Dict]) -> None:
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_entry(
    portfolio_value: float,
    weights: Dict[str, float],
    actions: Dict[str, str],
    dry_run: bool,
    note: str = "",
    run_date: Optional[str] = None,
) -> List[Dict]:
    """
    오늘자 실행 결과를 히스토리에 추가(같은 날짜로 재실행하면 덮어씀)하고
    README를 재생성한다. 갱신된 전체 히스토리를 반환한다.
    """
    run_date = run_date or date_cls.today().isoformat()
    history = [h for h in load_history() if h["date"] != run_date]

    prior_value = history[-1]["portfolio_value"] if history else None
    daily_return_pct = (
        round((portfolio_value / prior_value - 1) * 100, 3) if prior_value else None
    )

    history.append({
        "date": run_date,
        "portfolio_value": round(portfolio_value, 2),
        "daily_return_pct": daily_return_pct,
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "actions": actions,
        "dry_run": dry_run,
        "note": note,
    })
    history.sort(key=lambda h: h["date"])

    _save_history(history)
    update_readme(history)
    return history


def _render_table(history: List[Dict]) -> str:
    tickers = sorted({t for h in history for t in h["weights"] if t != "CASH"})
    header = "| 날짜 | 포트폴리오 가치 | 일간 수익률 | " + " | ".join(tickers) + " | 현금 | 비고 |"
    sep = "| --- | --- | --- |" + " --- |" * len(tickers) + " --- | --- |"
    lines = [header, sep]
    for h in history:
        ret = f"{h['daily_return_pct']:+.2f}%" if h["daily_return_pct"] is not None else "-"
        weight_cells = " | ".join(f"{h['weights'].get(t, 0.0):.1%}" for t in tickers)
        cash = h["weights"].get("CASH", 0.0)
        note = h.get("note") or ""
        if h.get("dry_run"):
            note = (note + " (dry-run)").strip()
        lines.append(
            f"| {h['date']} | ${h['portfolio_value']:,.2f} | {ret} | {weight_cells} "
            f"| {cash:.1%} | {note or '-'} |"
        )
    return "\n".join(lines)


def _render_chart(history: List[Dict]) -> Optional[str]:
    points = [h for h in history if h["daily_return_pct"] is not None]
    if len(points) < 2:
        return None

    dates = ", ".join(f'"{h["date"]}"' for h in points)
    returns = [h["daily_return_pct"] for h in points]
    returns_str = ", ".join(f"{r:.2f}" for r in returns)
    y_min = min(min(returns), 0) - 1
    y_max = max(max(returns), 0) + 1

    return "\n".join([
        "```mermaid",
        "xychart-beta",
        '    title "RL 포트폴리오 일간 수익률 (%)"',
        f"    x-axis [{dates}]",
        f'    y-axis "일간 수익률 (%)" {y_min:.1f} --> {y_max:.1f}',
        f"    bar [{returns_str}]",
        f"    line [{returns_str}]",
        "```",
    ])


def render_section(history: List[Dict]) -> str:
    """README에 그대로 끼워 넣을 마크다운(표 + 그래프)을 만든다.
    순수 함수 - 파일 I/O 없이 테스트/미리보기 가능."""
    if not history:
        return (
            "_아직 기록된 실행이 없습니다. `python -m rl.daily_run`을 실행하면 "
            "자동으로 채워집니다._"
        )

    # 주의: 마커 문자열 자체를 본문에 그대로 넣으면 다음 재생성 때 split이
    # 본문 속 마커에서 끊겨 README가 깨진다 - 이름만 백틱으로 표기한다.
    lines = [
        "실행할 때마다 이 표/그래프가 자동으로 갱신됩니다 (`rl/daily_run.py`가",
        "`rl/daily_run_history.json`에 결과를 추가하고 이 구간을 재생성합니다 -",
        "수동으로 이 마커(`RL_DAILY_LOG_START`/`_END`) 사이를 직접 편집하지",
        "마세요, 다음 실행 때 덮어써집니다). 드리프트 임계값 초과로 리밸런싱을",
        "건너뛴 실행은 기록하지 않습니다.",
        "",
    ]
    chart = _render_chart(history)
    if chart:
        lines += [chart, ""]
    lines.append(_render_table(history))
    return "\n".join(lines)


def update_readme(history: Optional[List[Dict]] = None) -> None:
    history = history if history is not None else load_history()
    with open(README_PATH, encoding="utf-8") as f:
        content = f.read()

    if README_LOG_START not in content or README_LOG_END not in content:
        print(f"경고: README.md에 {README_LOG_START}/{README_LOG_END} 마커가 없어 자동 갱신 건너뜀")
        return

    before, rest = content.split(README_LOG_START, 1)
    _, after = rest.split(README_LOG_END, 1)
    section = render_section(history)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(f"{before}{README_LOG_START}\n{section}\n{README_LOG_END}{after}")
