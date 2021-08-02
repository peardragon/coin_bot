import ccxt
import pyupbit

import private_key as key
import numpy as np
import min_algorithm


class CoinTrader:

    def __init__(self, mode):
        if mode == 0:
            access = key.bitmex_test_id
            secret = key.bitmex_test_secret
            self.exchange = ccxt.bitmex({
                'apiKey': access,
                'secret': secret,
            })
            if 'test' in self.exchange.urls:
                self.exchange.urls['api'] = self.exchange.urls['test']  # ←----- switch the base URL to testnet
        elif mode == 1:
            access = key.bitmex_id
            secret = key.bitmex_secret
            self.exchange = ccxt.bitmex({
                'apiKey': access,
                'secret': secret,
            })
        else:
            print('Error Occur')

    def get_current_data(self, limit):
        get_current_db = self.exchange.fetch_ohlcv('BTC/USD', '1m', limit=limit)    # list
        get_current_balance_avail = self.exchange.fetch_balance()['BTC']['free']
        return get_current_db, get_current_balance_avail

    def order_maker(self, current_ohlcv, current_available, algo, ratio=0.1, leverage=10):

        data = np.array(current_ohlcv)
        # ohlcv  /  db : olhcv : db 형식으로 바꾸기
        data[:, [1, 2]] = data[:, [2, 1]]
        # open low high close volume
        current_ohlcv = data.tolist()
        decision = algo.decision(current_ohlcv)

        orderbook = self.exchange.fetch_order_book('BTC/USD')
        bid = orderbook['bids'][0][0] if len(orderbook['bids']) > 0 else None
        ask = orderbook['asks'][0][0] if len(orderbook['asks']) > 0 else None

        # bid 는 주식에 매수자가 주식에 기꺼이 지불하려고 하는 최대 금액
        # - 즉각적인 가능 판매가
        # ask 는 주식 소유자가 판매하고자 하는 최소 금액
        # - 즉각적인 가능 구입가
        # ask 는 bid 보다 약간 높음.
        # The price : 마지막 매매가 /  ask or buy 둘 다 가능

        if bid is None or ask is None:
            decision = 'stay'

        if decision == 'buy':
            price = ask
        elif decision == 'sell':
            price = bid
        else:
            price = None

        amount = current_available * ratio * price if price is not None else 0

        if decision == 'buy':
            order = self.exchange.create_order('BTC/USD', 'limit', 'buy', amount, price, {'leverage': leverage})
        elif decision == 'sell':
            order = self.exchange.create_order('BTC/USD', 'limit', 'sell', amount, price, {'leverage': leverage})
        elif decision == 'stay':
            order = None
            pass

        print(f"Decision : {decision}")

        return order


if __name__ == '__main__':
    # trader = Trader()
    # print(".")
    from time import sleep
    coin_trader = CoinTrader(mode=0)
    algo = min_algorithm.VolatilityBreakOutStrategy()
    n = algo.observed_rows_num
    while True:
        sleep(5)
        db, avail = coin_trader.get_current_data(n)
        _order = coin_trader.order_maker(db, avail, algo, ratio=0.1, leverage=25)
        if _order is not None:
            if _order['status'] == 'closed':
                total = coin_trader.exchange.fetch_balance()['BTC']['total']
                print(f"total coin balance : {total}")
                print("----------------")
                continue
        else:
            print("stay")
            print("----------------")
