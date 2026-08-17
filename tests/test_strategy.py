import sys
import os
import csv
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import broker
import config
import strategy


@pytest.fixture(autouse=True)
def dry_run_by_default(monkeypatch):
    monkeypatch.setattr(config, "TOSS_LIVE_TRADING", False)


def test_decide_action_buy():
    assert strategy.decide_action(0.5) == "BUY"


def test_decide_action_sell():
    assert strategy.decide_action(-0.5) == "SELL"


def test_decide_action_hold():
    assert strategy.decide_action(0.1) == "HOLD"
    assert strategy.decide_action(config.BUY_THRESHOLD) == "HOLD"
    assert strategy.decide_action(config.SELL_THRESHOLD) == "HOLD"


def test_load_scores_watchlist_overrides_universe(tmp_path):
    universe_csv = tmp_path / "universe_analyst_scores_2026-01-01.csv"
    with open(universe_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "analyst_score"])
        writer.writeheader()
        writer.writerow({"ticker": "AAPL", "analyst_score": 0.2})
        writer.writerow({"ticker": "XOM", "analyst_score": 0.1})

    watchlist_csv = tmp_path / "signals_2026-01-01.csv"
    with open(watchlist_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "combined_score"])
        writer.writeheader()
        writer.writerow({"ticker": "AAPL", "combined_score": 0.9})

    scores = strategy.load_scores(output_dir=str(tmp_path))
    assert scores["AAPL"] == 0.9  # WATCHLIST(combined_score)가 우선
    assert scores["XOM"] == 0.1


def test_run_buy_flow_sizes_position_from_buying_power(monkeypatch):
    monkeypatch.setattr(broker, "get_holdings", lambda client: {"result": {"items": []}})
    monkeypatch.setattr(broker, "get_buying_power", lambda client, currency="USD": {"result": {"cashBuyingPower": "1000"}})

    healthy_metrics = {"netProfitMarginTTM": 10, "roeTTM": 15, "totalDebt/totalEquityAnnual": 1.0}
    monkeypatch.setattr(strategy.dp, "fetch_basic_financials", lambda symbol, key: healthy_metrics)

    with patch("broker.create_order") as mock_create:
        mock_create.return_value = {"dry_run": True}
        client = MagicMock()
        results = strategy.run(client, scores={"AAPL": 0.5})

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["symbol"] == "AAPL"
    assert call_kwargs["side"] == "BUY"
    assert call_kwargs["order_amount"] == 50.0  # 1000 * POSITION_SIZE_PCT(0.05)
    assert results[0]["action"] == "BUY"


def test_run_skips_buy_when_already_held(monkeypatch):
    monkeypatch.setattr(broker, "get_holdings", lambda client: {"result": {"items": [{"symbol": "AAPL", "quantity": "3"}]}})

    with patch("broker.create_order") as mock_create:
        client = MagicMock()
        results = strategy.run(client, scores={"AAPL": 0.5})

    mock_create.assert_not_called()
    assert results[0]["action"] == "SKIP_ALREADY_HELD"


def test_run_skips_buy_when_financially_unhealthy(monkeypatch):
    monkeypatch.setattr(broker, "get_holdings", lambda client: {"result": {"items": []}})
    unhealthy_metrics = {"netProfitMarginTTM": -5, "roeTTM": -10, "totalDebt/totalEquityAnnual": 3.0}
    monkeypatch.setattr(strategy.dp, "fetch_basic_financials", lambda symbol, key: unhealthy_metrics)

    with patch("broker.create_order") as mock_create:
        client = MagicMock()
        results = strategy.run(client, scores={"AAPL": 0.5})

    mock_create.assert_not_called()
    assert results[0]["action"] == "SKIP_UNHEALTHY_FINANCIALS"


def test_run_skips_buy_when_no_buying_power(monkeypatch):
    monkeypatch.setattr(broker, "get_holdings", lambda client: {"result": {"items": []}})
    monkeypatch.setattr(broker, "get_buying_power", lambda client, currency="USD": {"result": {"cashBuyingPower": "0"}})
    healthy_metrics = {"netProfitMarginTTM": 10, "roeTTM": 15, "totalDebt/totalEquityAnnual": 1.0}
    monkeypatch.setattr(strategy.dp, "fetch_basic_financials", lambda symbol, key: healthy_metrics)

    with patch("broker.create_order") as mock_create:
        client = MagicMock()
        results = strategy.run(client, scores={"AAPL": 0.5})

    mock_create.assert_not_called()
    assert results[0]["action"] == "SKIP_NO_BUYING_POWER"


def test_run_sell_flow_sells_full_sellable_quantity(monkeypatch):
    monkeypatch.setattr(broker, "get_holdings", lambda client: {"result": {"items": [{"symbol": "AAPL", "quantity": "3"}]}})
    monkeypatch.setattr(broker, "get_sellable_quantity", lambda client, symbol: {"result": {"sellableQuantity": "7"}})

    with patch("broker.create_order") as mock_create:
        mock_create.return_value = {"dry_run": True}
        client = MagicMock()
        results = strategy.run(client, scores={"AAPL": -0.5})

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["side"] == "SELL"
    assert call_kwargs["quantity"] == 7
    assert results[0]["action"] == "SELL"


def test_run_skips_sell_when_not_held(monkeypatch):
    monkeypatch.setattr(broker, "get_holdings", lambda client: {"result": {"items": []}})

    with patch("broker.create_order") as mock_create:
        client = MagicMock()
        results = strategy.run(client, scores={"AAPL": -0.5})

    mock_create.assert_not_called()
    assert results[0]["action"] == "SKIP_NOT_HELD"


def test_run_holds_when_score_in_neutral_range(monkeypatch):
    monkeypatch.setattr(broker, "get_holdings", lambda client: {"result": {"items": []}})

    with patch("broker.create_order") as mock_create:
        client = MagicMock()
        results = strategy.run(client, scores={"AAPL": 0.1})

    mock_create.assert_not_called()
    assert results[0]["action"] == "HOLD"


def test_run_passes_confirm_matching_live_trading_flag(monkeypatch):
    monkeypatch.setattr(config, "TOSS_LIVE_TRADING", True)
    monkeypatch.setattr(broker, "get_holdings", lambda client: {"result": {"items": [{"symbol": "AAPL", "quantity": "3"}]}})
    monkeypatch.setattr(broker, "get_sellable_quantity", lambda client, symbol: {"result": {"sellableQuantity": "3"}})

    with patch("broker.create_order") as mock_create:
        mock_create.return_value = {"orderId": "abc"}
        client = MagicMock()
        strategy.run(client, scores={"AAPL": -0.5})

    assert mock_create.call_args.kwargs["confirm"] is True
