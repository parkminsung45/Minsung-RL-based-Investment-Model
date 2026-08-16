import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl import daily_log


def _entry(date, portfolio_value, daily_return_pct, weights, actions=None, dry_run=True, note=""):
    return {
        "date": date,
        "portfolio_value": portfolio_value,
        "daily_return_pct": daily_return_pct,
        "weights": weights,
        "actions": actions or {},
        "dry_run": dry_run,
        "note": note,
    }


def test_render_section_empty_history():
    section = daily_log.render_section([])
    assert "아직 기록된 실행이 없습니다" in section


def test_render_table_includes_all_tickers_and_cash():
    history = [
        _entry("2026-08-10", 10000.0, None, {"AAPL": 0.5, "MSFT": 0.3, "CASH": 0.2}),
    ]
    table = daily_log._render_table(history)

    assert "AAPL" in table and "MSFT" in table
    assert "50.0%" in table  # AAPL 비중
    assert "20.0%" in table  # 현금 비중
    assert "$10,000.00" in table


def test_render_table_marks_dry_run_entries():
    history = [_entry("2026-08-10", 10000.0, None, {"AAPL": 1.0, "CASH": 0.0}, dry_run=True)]
    table = daily_log._render_table(history)
    assert "dry-run" in table


def test_render_chart_none_when_fewer_than_two_return_points():
    history = [_entry("2026-08-10", 10000.0, None, {"AAPL": 1.0, "CASH": 0.0})]
    assert daily_log._render_chart(history) is None


def test_render_chart_includes_dates_and_returns():
    history = [
        _entry("2026-08-10", 10000.0, None, {"AAPL": 1.0, "CASH": 0.0}),
        _entry("2026-08-11", 10100.0, 1.0, {"AAPL": 1.0, "CASH": 0.0}),
        _entry("2026-08-12", 9999.0, -1.0, {"AAPL": 1.0, "CASH": 0.0}),
    ]
    chart = daily_log._render_chart(history)

    assert chart is not None
    assert "xychart-beta" in chart
    assert '"2026-08-11"' in chart
    assert '"2026-08-12"' in chart
    assert "1.00" in chart and "-1.00" in chart


def test_append_entry_computes_return_relative_to_prior_day(tmp_path, monkeypatch):
    history_path = tmp_path / "daily_run_history.json"
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        f"intro\n{daily_log.README_LOG_START}\nold\n{daily_log.README_LOG_END}\noutro\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(daily_log, "HISTORY_PATH", str(history_path))
    monkeypatch.setattr(daily_log, "README_PATH", str(readme_path))

    daily_log.append_entry(
        portfolio_value=10000.0,
        weights={"AAPL": 1.0, "CASH": 0.0},
        actions={"AAPL": "BUY"},
        dry_run=True,
        run_date="2026-08-10",
    )
    history = daily_log.append_entry(
        portfolio_value=10100.0,
        weights={"AAPL": 1.0, "CASH": 0.0},
        actions={"AAPL": "HOLD"},
        dry_run=True,
        run_date="2026-08-11",
    )

    assert history[0]["daily_return_pct"] is None
    assert history[1]["daily_return_pct"] == 1.0

    readme_content = readme_path.read_text(encoding="utf-8")
    assert "intro" in readme_content and "outro" in readme_content
    assert "old" not in readme_content
    assert "2026-08-11" in readme_content
    # 수익률이 있는 지점이 1개뿐이면(최초 기록엔 전일 대비 기준이 없음) 그래프는
    # 아직 그리지 않는다 - test_render_chart_includes_dates_and_returns에서 별도 검증.
    assert "xychart-beta" not in readme_content


def test_append_entry_overwrites_same_date(tmp_path, monkeypatch):
    history_path = tmp_path / "daily_run_history.json"
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        f"{daily_log.README_LOG_START}\n{daily_log.README_LOG_END}\n", encoding="utf-8"
    )
    monkeypatch.setattr(daily_log, "HISTORY_PATH", str(history_path))
    monkeypatch.setattr(daily_log, "README_PATH", str(readme_path))

    daily_log.append_entry(
        portfolio_value=10000.0, weights={"CASH": 1.0}, actions={}, dry_run=True, run_date="2026-08-10"
    )
    history = daily_log.append_entry(
        portfolio_value=10500.0, weights={"CASH": 1.0}, actions={}, dry_run=True, run_date="2026-08-10"
    )

    assert len(history) == 1
    assert history[0]["portfolio_value"] == 10500.0


def test_rendered_section_does_not_contain_raw_markers(tmp_path, monkeypatch):
    # 본문에 마커 문자열이 그대로 들어가면 다음 재생성 때 split이 본문 속
    # 마커에서 끊겨 README가 깨진다 - 연속 재생성 후에도 마커가 정확히
    # 한 쌍만 남아야 한다.
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        f"intro\n{daily_log.README_LOG_START}\n{daily_log.README_LOG_END}\noutro\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(daily_log, "README_PATH", str(readme_path))

    history = [_entry("2026-08-10", 10000.0, None, {"AAPL": 1.0, "CASH": 0.0})]
    daily_log.update_readme(history)
    daily_log.update_readme(history)  # 두 번째 재생성에서 깨지지 않아야 함

    content = readme_path.read_text(encoding="utf-8")
    assert content.count(daily_log.README_LOG_START) == 1
    assert content.count(daily_log.README_LOG_END) == 1
    assert "intro" in content and "outro" in content


def test_update_readme_warns_without_crashing_when_markers_missing(tmp_path, monkeypatch, capsys):
    readme_path = tmp_path / "README.md"
    readme_path.write_text("no markers here\n", encoding="utf-8")
    monkeypatch.setattr(daily_log, "README_PATH", str(readme_path))

    daily_log.update_readme([])

    assert "no markers here\n" == readme_path.read_text(encoding="utf-8")
    assert "마커가 없어" in capsys.readouterr().out
