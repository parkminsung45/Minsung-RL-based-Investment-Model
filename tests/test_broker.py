import sys
import os
import time
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import broker


@pytest.fixture(autouse=True)
def dry_run_by_default(monkeypatch):
    monkeypatch.setattr(config, "TOSS_LIVE_TRADING", False)


def _fake_token_response(access_token="tok-1", expires_in=86400):
    resp = MagicMock()
    resp.json.return_value = {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": expires_in,
    }
    resp.raise_for_status.return_value = None
    return resp


def test_fetch_access_token_sends_client_credentials_grant():
    with patch("broker.requests.post") as mock_post:
        mock_post.return_value = _fake_token_response()
        token = broker.fetch_access_token("id-1", "secret-1")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.args[0] == broker.TOKEN_URL
        assert call_args.kwargs["data"] == {
            "grant_type": "client_credentials",
            "client_id": "id-1",
            "client_secret": "secret-1",
        }
        assert token["access_token"] == "tok-1"


def test_request_without_account_raises_when_account_required():
    client = broker.TossClient("id", "secret")
    with patch.object(client, "_ensure_token", return_value="tok"):
        with pytest.raises(RuntimeError):
            client._headers(with_account=True)


def test_request_includes_account_header_after_set_account():
    client = broker.TossClient("id", "secret")
    client.set_account(12345)
    with patch.object(client, "_ensure_token", return_value="tok"):
        headers = client._headers(with_account=True)
    assert headers["X-Tossinvest-Account"] == "12345"
    assert headers["Authorization"] == "Bearer tok"


def test_request_reuses_cached_token_until_near_expiry():
    client = broker.TossClient("id", "secret")
    with patch("broker.fetch_access_token") as mock_fetch:
        mock_fetch.return_value = {"access_token": "tok-a", "expires_at": time.time() + 3600}
        first = client._ensure_token()
        second = client._ensure_token()
    assert first == "tok-a"
    assert second == "tok-a"
    mock_fetch.assert_called_once()


def test_create_order_dry_run_does_not_call_client():
    client = MagicMock()
    result = broker.create_order(
        client, symbol="AAPL", side="BUY", order_type="MARKET", quantity=1
    )
    client.request.assert_not_called()
    assert result["dry_run"] is True
    assert result["would_send"]["symbol"] == "AAPL"
    assert result["would_send"]["side"] == "BUY"


def test_create_order_rejects_invalid_side():
    client = MagicMock()
    with pytest.raises(ValueError):
        broker.create_order(client, symbol="AAPL", side="HOLD", order_type="MARKET", quantity=1)


def test_create_order_rejects_invalid_order_type():
    client = MagicMock()
    with pytest.raises(ValueError):
        broker.create_order(client, symbol="AAPL", side="BUY", order_type="STOP", quantity=1)


def test_create_order_requires_exactly_one_of_quantity_or_amount():
    client = MagicMock()
    with pytest.raises(ValueError):
        broker.create_order(client, symbol="AAPL", side="BUY", order_type="MARKET")
    with pytest.raises(ValueError):
        broker.create_order(
            client, symbol="AAPL", side="BUY", order_type="MARKET",
            quantity=1, order_amount=100,
        )


def test_create_order_limit_requires_price():
    client = MagicMock()
    with pytest.raises(ValueError):
        broker.create_order(client, symbol="AAPL", side="BUY", order_type="LIMIT", quantity=1)


def test_create_order_live_without_confirm_raises(monkeypatch):
    monkeypatch.setattr(config, "TOSS_LIVE_TRADING", True)
    client = MagicMock()
    with pytest.raises(RuntimeError):
        broker.create_order(client, symbol="AAPL", side="BUY", order_type="MARKET", quantity=1)
    client.request.assert_not_called()


def test_create_order_live_with_confirm_calls_client(monkeypatch):
    monkeypatch.setattr(config, "TOSS_LIVE_TRADING", True)
    client = MagicMock()
    client.request.return_value = {"orderId": "abc123"}

    result = broker.create_order(
        client, symbol="AAPL", side="BUY", order_type="MARKET", quantity=1, confirm=True
    )

    client.request.assert_called_once()
    call_args = client.request.call_args
    assert call_args.args[0] == "POST"
    assert call_args.args[1] == "/api/v1/orders"
    assert call_args.kwargs["with_account"] is True
    assert call_args.kwargs["json"]["symbol"] == "AAPL"
    assert result == {"orderId": "abc123"}


def test_cancel_order_dry_run_does_not_call_client():
    client = MagicMock()
    result = broker.cancel_order(client, order_id="abc123")
    client.request.assert_not_called()
    assert result["dry_run"] is True


def test_cancel_order_live_without_confirm_raises(monkeypatch):
    monkeypatch.setattr(config, "TOSS_LIVE_TRADING", True)
    client = MagicMock()
    with pytest.raises(RuntimeError):
        broker.cancel_order(client, order_id="abc123")
    client.request.assert_not_called()


def test_modify_order_dry_run_does_not_call_client():
    client = MagicMock()
    result = broker.modify_order(client, order_id="abc123", price=150)
    client.request.assert_not_called()
    assert result["dry_run"] is True
    assert result["would_send"]["price"] == 150


def test_create_order_rejects_krx_symbols():
    # 미국 주식 전용 계좌(2026-08-17 결정) - KRX 6자리 숫자 코드는 무조건 차단.
    client = MagicMock()
    with pytest.raises(ValueError, match="KRX"):
        broker.create_order(
            client, symbol="005930", side="BUY", order_type="MARKET", quantity=1
        )
    client.request.assert_not_called()
