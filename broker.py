"""
토스증권 Open API 클라이언트: 인증, 계좌/자산 조회, 주문 생성/조회/취소/정정.
문서: https://openapi.tossinvest.com/openapi-docs/latest

*** 중요: 이 API는 샌드박스/모의투자 환경이 없다. ***
주문 생성 요청은 즉시 실제 계좌·실제 자금에 반영된다.

주문 관련 함수의 이중 안전장치:
  1. config.TOSS_LIVE_TRADING이 False(기본값)인 동안은 실제로 주문을
     보내지 않고, 보낼 요청 내용만 출력/반환한다 (dry-run).
  2. TOSS_LIVE_TRADING=true로 실거래를 켠 상태에서도, 각 함수 호출 시
     confirm=True를 명시하지 않으면 실행을 거부한다.
  두 조건을 모두 충족해야 실제 주문이 나간다.
"""
import time
from typing import Any, Dict, Optional

import requests

import config

BASE_URL = "https://openapi.tossinvest.com"
TOKEN_URL = f"{BASE_URL}/oauth2/token"

# 토큰 만료 시각보다 이만큼 일찍 갱신한다 (초).
TOKEN_REFRESH_MARGIN_SEC = 30


def fetch_access_token(client_id: str, client_secret: str) -> Dict:
    """
    access_token과 만료 시각(epoch seconds)을 반환한다.
    새 토큰을 발급받으면 이전에 발급된 토큰은 즉시 무효화된다. refresh token은
    제공되지 않으므로, 만료가 임박하면 다시 client_credentials로 새 토큰을 받는다.
    """
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "access_token": data["access_token"],
        "expires_at": time.time() + data.get("expires_in", 0),
    }


class TossClient:
    def __init__(self, client_id: str, client_secret: str, account_seq: Optional[int] = None):
        self._client_id = client_id
        self._client_secret = client_secret
        self.account_seq = account_seq
        self._token: Optional[Dict] = None

    def set_account(self, account_seq: int) -> None:
        """get_accounts()로 조회한 accountSeq를 지정한다."""
        self.account_seq = account_seq

    def _ensure_token(self) -> str:
        if self._token is None or time.time() >= self._token["expires_at"] - TOKEN_REFRESH_MARGIN_SEC:
            self._token = fetch_access_token(self._client_id, self._client_secret)
        return self._token["access_token"]

    def _headers(self, with_account: bool) -> Dict:
        headers = {"Authorization": f"Bearer {self._ensure_token()}"}
        if with_account:
            if self.account_seq is None:
                raise RuntimeError(
                    "account_seq가 설정되지 않았습니다. "
                    "get_accounts() 조회 후 client.set_account(accountSeq)를 호출하세요."
                )
            headers["X-Tossinvest-Account"] = str(self.account_seq)
        return headers

    def request(self, method: str, path: str, with_account: bool = False, **kwargs: Any) -> Dict:
        resp = requests.request(
            method,
            f"{BASE_URL}{path}",
            headers=self._headers(with_account),
            timeout=15,
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}


# ---------------------------------------------------------------------------
# 계좌/자산 조회
# ---------------------------------------------------------------------------

def get_accounts(client: TossClient) -> Dict:
    """
    종합매매(BROKERAGE) 계좌 목록을 조회한다.
    응답에 포함된 accountSeq를 client.set_account()에 넘겨야
    이후 계좌 관련 API(보유종목, 주문 등)를 호출할 수 있다.
    """
    return client.request("GET", "/api/v1/accounts")


def get_holdings(client: TossClient) -> Dict:
    """
    실계좌 검증 완료(2026-08-17): 응답은 {"result": {..., "items": [{"symbol",
    "quantity"(str), "lastPrice"(str), ...}]}} 형태이며 수치는 전부 문자열이다.
    """
    return client.request("GET", "/api/v1/holdings", with_account=True)


def get_buying_power(client: TossClient, currency: str = "USD") -> Dict:
    """
    실계좌 검증 완료(2026-08-17): currency 쿼리 파라미터가 필수이며(KRW/USD),
    응답은 {"result": {"currency", "cashBuyingPower"(str)}} 형태다.
    """
    return client.request(
        "GET", "/api/v1/buying-power", with_account=True, params={"currency": currency}
    )


def get_sellable_quantity(client: TossClient, symbol: str) -> Dict:
    return client.request(
        "GET", "/api/v1/sellable-quantity", with_account=True, params={"symbol": symbol}
    )


# ---------------------------------------------------------------------------
# 주문 생성/조회/취소/정정
# ---------------------------------------------------------------------------

def create_order(
    client: TossClient,
    symbol: str,
    side: str,
    order_type: str,
    quantity: Optional[float] = None,
    order_amount: Optional[float] = None,
    price: Optional[float] = None,
    time_in_force: str = "DAY",
    confirm_high_value_order: bool = False,
    confirm: bool = False,
) -> Dict:
    """
    symbol: 종목 심볼 (KRX 6자리 숫자 / US 티커)
    side: "BUY" 또는 "SELL"
    order_type: "LIMIT" 또는 "MARKET"
    quantity / order_amount: 둘 중 정확히 하나만 지정 (수량 기반 vs 금액 기반 주문)
    price: order_type이 LIMIT일 때 필수
    confirm_high_value_order: 1억원 이상 주문 시 착오주문 방지용 확인 플래그
    confirm: True로 명시해야 실거래 모드(config.TOSS_LIVE_TRADING=true)에서 실제 전송됨
    """
    # 이 계좌는 미국 주식 전용으로 운용한다(2026-08-17 결정). KRX 종목코드는
    # 6자리 숫자라 형태로 구분 가능 - 국내 주식 주문은 여기서 무조건 차단한다.
    if symbol.isdigit():
        raise ValueError(
            f"국내(KRX) 종목 주문은 차단되어 있습니다: {symbol!r}. "
            "이 시스템은 미국 주식 전용입니다."
        )
    if side not in ("BUY", "SELL"):
        raise ValueError("side는 'BUY' 또는 'SELL'이어야 합니다.")
    if order_type not in ("LIMIT", "MARKET"):
        raise ValueError("order_type은 'LIMIT' 또는 'MARKET'이어야 합니다.")
    if (quantity is None) == (order_amount is None):
        raise ValueError("quantity와 order_amount 중 정확히 하나만 지정해야 합니다.")
    if order_type == "LIMIT" and price is None:
        raise ValueError("LIMIT 주문은 price가 필요합니다.")

    body = {
        "symbol": symbol,
        "side": side,
        "orderType": order_type,
        "timeInForce": time_in_force,
        "confirmHighValueOrder": confirm_high_value_order,
    }
    if quantity is not None:
        body["quantity"] = quantity
    if order_amount is not None:
        body["orderAmount"] = order_amount
    if price is not None:
        body["price"] = price

    if not config.TOSS_LIVE_TRADING:
        print(f"[DRY RUN] 실제 주문을 보내지 않았습니다: {body}")
        return {"dry_run": True, "would_send": body}

    if not confirm:
        raise RuntimeError(
            "TOSS_LIVE_TRADING=true 상태입니다. 실제 주문을 보내려면 "
            "create_order(..., confirm=True)를 명시적으로 호출하세요."
        )

    return client.request("POST", "/api/v1/orders", with_account=True, json=body)


def cancel_order(client: TossClient, order_id: str, confirm: bool = False) -> Dict:
    if not config.TOSS_LIVE_TRADING:
        print(f"[DRY RUN] 주문 취소를 보내지 않았습니다: order_id={order_id}")
        return {"dry_run": True, "order_id": order_id}

    if not confirm:
        raise RuntimeError(
            "TOSS_LIVE_TRADING=true 상태입니다. 실제 취소를 보내려면 "
            "cancel_order(..., confirm=True)를 명시적으로 호출하세요."
        )

    return client.request("POST", f"/api/v1/orders/{order_id}/cancel", with_account=True)


def modify_order(
    client: TossClient,
    order_id: str,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    confirm: bool = False,
) -> Dict:
    """
    국내 주식: quantity(양의 정수) 정정 가능.
    미국 주식: quantity 정정 불가, price만 변경 가능.
    """
    body = {}
    if quantity is not None:
        body["quantity"] = quantity
    if price is not None:
        body["price"] = price

    if not config.TOSS_LIVE_TRADING:
        print(f"[DRY RUN] 주문 정정을 보내지 않았습니다: order_id={order_id}, {body}")
        return {"dry_run": True, "order_id": order_id, "would_send": body}

    if not confirm:
        raise RuntimeError(
            "TOSS_LIVE_TRADING=true 상태입니다. 실제 정정을 보내려면 "
            "modify_order(..., confirm=True)를 명시적으로 호출하세요."
        )

    return client.request("POST", f"/api/v1/orders/{order_id}/modify", with_account=True, json=body)


def get_orders(client: TossClient) -> Dict:
    return client.request("GET", "/api/v1/orders", with_account=True)


def get_order(client: TossClient, order_id: str) -> Dict:
    return client.request("GET", f"/api/v1/orders/{order_id}", with_account=True)
