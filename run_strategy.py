"""
strategy.py의 매매 전략을 실제 계좌에 대해 실행한다.
main.py / scan_universe.py가 만든 output/signals_*.csv,
output/universe_analyst_scores_*.csv의 점수를 사용한다.

config.TOSS_LIVE_TRADING=false(기본값)인 동안은 broker.create_order()가
실제 주문을 보내지 않고 dry-run으로만 동작한다.

실행 방법: python run_strategy.py
"""
import broker
import config
import strategy


def run():
    if not config.TOSS_CLIENT_ID or not config.TOSS_CLIENT_SECRET:
        raise RuntimeError(".env 파일에 TOSS_CLIENT_ID와 TOSS_CLIENT_SECRET을 설정하세요.")

    client = broker.TossClient(config.TOSS_CLIENT_ID, config.TOSS_CLIENT_SECRET)

    # 실계좌 검증 완료(2026-08-17): 계좌 목록은 result 키 아래 리스트로 온다.
    account_list = broker.get_accounts(client).get("result") or []
    if not account_list:
        raise RuntimeError("조회된 계좌가 없습니다.")
    client.set_account(account_list[0]["accountSeq"])

    results = strategy.run(client)
    for r in results:
        print(r)
    return results


if __name__ == "__main__":
    run()
