import pandas as pd
import random
import lightgbm_model
from sklearn.preprocessing import MinMaxScaler

# 매수 주문 시 정산금액 = 체결금액(주문수량 x 주문가격) + 거래수수료
# 예시) 1BTC 를 10,000,000원에 매수(거래수수료 0.139%) 시 내 계정에 1BTC 반영, 10,013,900원 차감
# 매수시, 원하는 만큼 정확한 양이 매수됨. 거래시 수수료 발생 .

# 매도 주문 시 정산금액 = 체결금액(주문수량 x 주문가격) – 거래수수료
# 예시) 1BTC 를 10,000,000원에 매도(거래수수료 0.139%) 시 내 계정에 9,986,100원 반영, 1 BTC 차감
# 매도시, 원하는 만큼 정확한 양이 매도됨. 거래시 수수료 발생 .
# fee = {"KRW": 0.05, "BTC": 0.25, "USDT": 0.05}


#input data : ndarry
# [[ open low high close volume ],
#  [ open low high close volume ],
#  [ open low high close volume ],
#  ...
#  [ open low high close volume ]]


# Limit order - 0.35
class Algorithm:
    def __init__(self):
        self.__name__ = "algo1"
        self.observed_rows_num = 5
        self.execute_ratio_buy = 0.1
        self.execute_ratio_sell = 0.5
        random.seed(1)

        # 필수 변수

    def decision(self, data: pd.DataFrame):
        decision = random.choice(['buy', 'sell', 'stay', "close"])
        current_price = data.tail(1).close.item()
        limit_price_lower = current_price * 0.95
        limit_price_upper = current_price * 1.05

        return {'decision': decision, 'limit_high': limit_price_upper, 'limit_low': limit_price_lower}

import pickle
import numpy as np

def restore_model(filename):
    with open(filename, 'rb') as model:
        loaded_model = pickle.load(model)
    return loaded_model


class LGBMAlgorithm:
    def __init__(self, config: dict):
        self.__name__ ="LGBM_algo"
        self.observed_rows_num = config["interval"]
        self.execute_ratio_buy = 0.5
        self.execute_ratio_sell = 1
        self.model = restore_model(f'./lgbm_model/{config["time"]}/'
                                   f'optimized_model_interval_{str(config["interval"])}_rate_{str(config["rate"])}.sav')
        self.rate = config["rate"]
        self.num = 0

    def decision(self, data):
        # open low high close volume
        X = []
        curr = data[:, [0, 2, 4]]
        scaled_data = MinMaxScaler().fit_transform(X=curr)
        X.append(scaled_data.flatten())
        X = np.asarray(X, dtype=float)

        # X = np.asarray(scaled_data.flatten(), dtype=float).reshape(1,-1)
        # pred_proba = self.model.predict_proba(X)
        # pred = [i > 0.8 for i in pred_proba[:, 1]]
        pred = self.model.predict(X)

        if pred[0] == True:
            decision = 'buy'
        else:
            decision = "stay"
        # open low high close volume
        current_price = data[-1, 3]
        limit_price_lower = current_price * (1-self.rate/100)
        limit_price_upper = current_price * (1+self.rate/100)

        self.num += 1
        return {'decision': decision, 'limit_high': limit_price_upper, 'limit_low': limit_price_lower}


class LimitAlgorithm:
    def __init__(self):
        self.__name__ = "algo2"
        self.observed_rows_num = 5
        self.execute_ratio_buy = 0.1
        self.execute_ratio_sell = 0.5
        random.seed(1)

        # 필수 변수

    def decision(self, data):
        decision = random.choice(['buy', 'stay', 'stay', 'stay', 'stay'])
        # open low high close volume
        current_price = data[-1, 3]
        limit_price_lower = current_price * 0.99
        limit_price_upper = current_price * 1.01

        return {'decision': decision, 'limit_high': limit_price_upper, 'limit_low': limit_price_lower}


class Algorithm1:
    def __init__(self):
        self.__name__ = "algo1"
        self.observed_rows_num = 30
        # 필수 변수

    def buy_algo(self, data: pd.DataFrame, balance_krw):
        n = self.observed_rows_num
        if float(balance_krw) * 0.2 > 5000:
            pass
        else:
            return "stay"
        current_data = float(data[-1:]["close"].item())
        before_data_average = data[0:n]['close'].mean()
        if current_data * (1 + 0.4 / 100) < before_data_average:  # 현재가가 이전가격보다 저평가 되어 있다면,
            return True
        else:
            return "stay"

    def sell_algo(self, data: pd.DataFrame, balance_ticker):
        n = self.observed_rows_num
        if float(balance_ticker) > 0:
            pass
        else:
            return "stay"
        current_data = float(data[-1:]["close"].item())
        before_data_average = data[0:n]['close'].mean()
        if current_data > before_data_average * (1 + 0.4 / 100):  # 현재가가 이전가격보다 고평가 되어 있다면.
            return True
        else:
            return "stay"


class UpperMomentumAlgo:

    def __init__(self, time_limit):
        self.__name__ = "algo1"
        self.observed_rows_num = 5
        self.buy_executed = False
        self.buy_executed_time = 0
        self.time_limit = time_limit

    def buy_algo(self, data: pd.DataFrame, balance_krw):
        n = self.observed_rows_num

        current_data = data[-1:]["close"].item()
        before_data = data[0:n]['close']
        lower_limit = data['low'].min()
        upper_limit = data['high'].max()

        is_upper_limit = upper_limit == current_data
        print(f"상한 {upper_limit} , 현재가 {current_data}")

        if is_upper_limit and not self.buy_executed:
            self.buy_executed = True
            return 'buy', current_data
        else:
            print(f"매수 여부 {self.buy_executed} 매수 후 진행 시간 :{self.buy_executed_time}분")
            return 'stay', None

    def sell_algo(self, data: pd.DataFrame, balance_ticker):
        n = self.observed_rows_num
        current_data = float(data[-1:]["close"].item())

        if self.buy_executed and self.buy_executed_time > self.time_limit:
            self.buy_executed_time = 0
            self.buy_executed = False
            return True, current_data
        else:
            if self.buy_executed:
                self.buy_executed_time += 1
            else:
                pass
            return "stay", None


class RandomAlgorithm:

    def __init__(self):
        self.__name__ = "algo1"
        self.observed_rows_num = 30

    def decision(self, data, balance_coin):
        import random
        choice = random.choice(['buy', 'sell', 'stay'])

        return choice


# ccxt 의 아웃풋은 list 형태이므로, simulation 부분에서 dataframe.values.tolist()로 list output을 받는다고 하자.
# 순서는 0: time 1: open 2: low 3: high 4: close 5: volume
# observed_rows_num 에 해당하는 개수의 데이터 ex) limit = 1 : 1개 / limit = 30 : 30개 (0~29 인덱스)


class VolatilityBreakOutStrategy:

    def __init__(self):
        self.__name__ = "volatility_break_out"
        self.observed_rows_num = 2
        self.taken_time = 0
        self.wait_time = 0
        self.buy_executed = False
        self.base_executed = False
        self.base_time = None
        self.position_close_time = None
        self.before_decision = None
        self.base_long = None
        self.base_short = None

    def decision(self, data: list, balance_coin=None):
        n = self.observed_rows_num
        before_data = data[-n]
        current_data = data[-1]
        current_price = current_data[1]
        if current_price is None:
            return 'stay'
        _range_ = before_data[3] - before_data[2]
        K = 0.5

        if self.buy_executed:
            self.taken_time += 1
            if self.taken_time > self.wait_time:
                if self.before_decision == 'buy':
                    decision = 'sell'
                    self.buy_executed = False
                    self.taken_time = 0
                    self.base_executed = False
                    print("Taken time: ", self.taken_time)
                    return decision
                elif self.before_decision == 'sell':
                    decision = 'buy'
                    self.buy_executed = False
                    self.taken_time = 0
                    self.base_executed = False
                    print("Taken time: ", self.taken_time)
                    return decision
            else:
                decision = 'stay'
                print("Taken time: ", self.taken_time)
                return decision

        if not self.base_executed:
            self.base_long = base_price_long = _range_ * K + current_price
            self.base_short = base_price_short = - _range_ * K + current_price
            self.base_time = before_data[0]
            self.base_executed = True

        if self.base_long < current_price:
            decision = self.before_decision = 'buy'
            self.buy_executed = True
            self.wait_time = self.taken_time
            self.taken_time = 0

        elif self.base_short > current_price:
            decision = self.before_decision = 'sell'
            self.buy_executed = True
            self.wait_time = self.taken_time
            self.taken_time = 0

        else:
            if self.before_decision == 'stay':
                self.taken_time += 1
                decision = 'stay'
            else:
                decision = self.before_decision = 'stay'
                self.buy_executed = False
                self.taken_time = 1

            print("Taken time: ", self.taken_time)

        return decision
