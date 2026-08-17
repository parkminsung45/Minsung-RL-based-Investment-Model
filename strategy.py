"""
signals_*.csv(WATCHLIST, combined_score) / universe_analyst_scores_*.csv
(그 외 종목, analyst_score)의 점수를 실제 매수/매도 판단으로 연결한다.

규칙:
  - score > config.BUY_THRESHOLD  -> 매수 후보
      -> 이미 보유 중이면 건너뜀
      -> data_pipeline.passes_financial_health()를 통과하지 못하면 건너뜀
         (재무 건전성 필터: 순이익률>0, ROE>0, 부채비율<2.0)
      -> 매수가능금액(buying power)의 config.POSITION_SIZE_PCT 만큼 시장가 매수
  - score < config.SELL_THRESHOLD -> 보유 중이면 전량 시장가 매도
  - 그 외                          -> 홀드 (아무 것도 하지 않음)

broker.create_order()의 기존 안전장치(config.TOSS_LIVE_TRADING=false 기본
드라이런)가 그대로 적용된다. confirm은 이 모듈이 config.TOSS_LIVE_TRADING과
동일한 값으로 넘겨준다 - 즉 "실거래를 켠다"는 결정 자체가 유일한 게이트이며,
그 결정을 내린 뒤에는 전략이 사람 개입 없이 실행될 수 있다.

응답 필드(buyingPower, sellableQuantity, holdings의 symbol 키 등)는 토스증권
API 신청이 아직 승인되지 않아 실제 자격증명으로 검증하지 못했다. 승인 후
실제 응답을 확인해 아래 후보 키 목록을 정리할 것.
"""
import csv
import glob
import os
from typing import Dict, List, Optional

import broker
import config
import data_pipeline as dp


def load_scores(output_dir: Optional[str] = None) -> Dict[str, float]:
    """
    가장 최근 universe_analyst_scores_*.csv(analyst_score)와
    signals_*.csv(WATCHLIST, combined_score)를 읽어 티커별 점수 하나로
    합친다. 같은 티커가 양쪽에 있으면 combined_score(뉴스+애널리스트)를
    우선한다.
    """
    output_dir = output_dir or config.OUTPUT_DIR
    scores: Dict[str, float] = {}

    universe_files = sorted(glob.glob(os.path.join(output_dir, "universe_analyst_scores_*.csv")))
    if universe_files:
        with open(universe_files[-1], newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scores[row["ticker"]] = float(row["analyst_score"])

    watchlist_files = sorted(glob.glob(os.path.join(output_dir, "signals_*.csv")))
    if watchlist_files:
        with open(watchlist_files[-1], newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scores[row["ticker"]] = float(row["combined_score"])

    return scores


def decide_action(score: float) -> str:
    if score > config.BUY_THRESHOLD:
        return "BUY"
    if score < config.SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def _extract_value(response: Dict, candidate_keys: List[str]) -> Optional[float]:
    """
    실계좌 검증 완료(2026-08-17): 응답 본문은 {"result": {...}}로 감싸져 있고
    수치는 문자열이다. result 안쪽을 먼저 보고, float()가 문자열도 처리한다.
    """
    if isinstance(response.get("result"), dict):
        response = response["result"]
    for key in candidate_keys:
        if key in response and response[key] is not None:
            return float(response[key])
    return None


def _holdings_items(client: broker.TossClient) -> List[Dict]:
    """실계좌 검증 완료(2026-08-17): 보유종목 리스트는 result.items에 있다."""
    holdings = broker.get_holdings(client)
    if not isinstance(holdings, dict):
        return []
    result = holdings.get("result", holdings)
    items = result.get("items") if isinstance(result, dict) else None
    return items if isinstance(items, list) else []


def _held_symbols(client: broker.TossClient) -> set:
    return {item["symbol"] for item in _holdings_items(client) if item.get("symbol")}


def run(client: broker.TossClient, scores: Optional[Dict[str, float]] = None) -> List[Dict]:
    scores = scores if scores is not None else load_scores()
    held_symbols = _held_symbols(client)
    confirm = config.TOSS_LIVE_TRADING

    results = []
    for symbol, score in scores.items():
        action = decide_action(score)

        if action == "BUY":
            if symbol in held_symbols:
                results.append({"symbol": symbol, "score": score, "action": "SKIP_ALREADY_HELD"})
                continue

            metrics = dp.fetch_basic_financials(symbol, config.FINNHUB_API_KEY)
            if not dp.passes_financial_health(
                metrics, config.MIN_NET_MARGIN, config.MIN_ROE, config.MAX_DEBT_TO_EQUITY
            ):
                results.append({"symbol": symbol, "score": score, "action": "SKIP_UNHEALTHY_FINANCIALS"})
                continue

            buying_power = broker.get_buying_power(client, currency="USD")
            available = _extract_value(buying_power, ["cashBuyingPower"])
            order_amount = round((available or 0) * config.POSITION_SIZE_PCT, 2)
            if order_amount <= 0:
                results.append({"symbol": symbol, "score": score, "action": "SKIP_NO_BUYING_POWER"})
                continue

            order = broker.create_order(
                client, symbol=symbol, side="BUY", order_type="MARKET",
                order_amount=order_amount, confirm=confirm,
            )
            results.append({"symbol": symbol, "score": score, "action": "BUY", "order": order})

        elif action == "SELL":
            if symbol not in held_symbols:
                results.append({"symbol": symbol, "score": score, "action": "SKIP_NOT_HELD"})
                continue

            sellable = broker.get_sellable_quantity(client, symbol)
            quantity = _extract_value(sellable, ["sellableQuantity", "quantity", "availableQuantity"])
            if not quantity:
                results.append({"symbol": symbol, "score": score, "action": "SKIP_NO_SELLABLE_QTY"})
                continue

            order = broker.create_order(
                client, symbol=symbol, side="SELL", order_type="MARKET",
                quantity=quantity, confirm=confirm,
            )
            results.append({"symbol": symbol, "score": score, "action": "SELL", "order": order})

        else:
            results.append({"symbol": symbol, "score": score, "action": "HOLD"})

    return results
