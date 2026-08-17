import os
import sys
from datetime import date
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl import daily_log, daily_run


def test_already_ran_today_true_when_last_entry_matches(monkeypatch):
    monkeypatch.setattr(daily_log, "load_history", lambda: [{"date": "2026-08-17"}])
    assert daily_run._already_ran_today("2026-08-17") is True


def test_already_ran_today_false_when_last_entry_is_older(monkeypatch):
    monkeypatch.setattr(daily_log, "load_history", lambda: [{"date": "2026-08-14"}])
    assert daily_run._already_ran_today("2026-08-17") is False


def test_already_ran_today_false_when_no_history(monkeypatch):
    monkeypatch.setattr(daily_log, "load_history", lambda: [])
    assert daily_run._already_ran_today("2026-08-17") is False


def test_run_skips_on_weekend_without_touching_broker():
    # 2026-08-15는 토요일 - 모델 로드/브로커 호출 전에 걸러져야 한다.
    saturday = date(2026, 8, 15)
    with patch("rl.daily_run.rl_strategy") as mock_rl_strategy:
        results = daily_run.run(today=saturday)
    mock_rl_strategy.load_model_and_meta.assert_not_called()
    assert results == []


def test_run_skips_when_already_ran_today(monkeypatch):
    # 2026-08-17은 월요일이라 주말 가드는 통과하고, 이미 실행 기록이 있어 스킵돼야 한다.
    monkeypatch.setattr(daily_log, "load_history", lambda: [{"date": "2026-08-17"}])
    with patch("rl.daily_run.rl_strategy") as mock_rl_strategy:
        results = daily_run.run(today=date(2026, 8, 17))
    mock_rl_strategy.load_model_and_meta.assert_not_called()
    assert results == []
