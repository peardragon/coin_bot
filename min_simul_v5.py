from mysql_func import *
from min_algorithm import *
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt

# 안정성 테스트
# (Future, DB init, Save, Limit)
# y y y y = y y n y 확인.


from functools import wraps


def time_checker(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        print(f'Success. {end - start} taken for {func.__name__}')
        return result

    return wrapper


class Simulator:

    #  시뮬레이터의 초기화 부
    #  시작 초기 자본 및 future 거래 여부 설정.
    def __init__(self, init_balance, init_coin_balance):

        # 기본 자금 설정
        self.init_balance = init_balance
        self.init_coin_balance = init_coin_balance
        self.leverage = 1

        # 기본 Setting
        self.init = True
        self.future = False
        self.save_per_row = True
        self.limit = False
        self.liquidation = False
        self.fee = 0.25 / 100

        # Algorithm 관련 변수
        self.algo = None
        self.observed_rows = 0
        self.buy_ratio = 1
        self.sell_ratio = 1

        # table 관련 변수
        self.db_start_time = None
        self.db_finish_time = None
        self.total_table_name = None
        self.transaction_table_name = None
        self.order_list_name = None
        self.simulator_start_time = None

        self.current_time = None
        self.simulated_time = None

        # Total DB 관련 변수
        self.total_order_db = None
        self.total_transaction_db = None

        self.total_order_data = None
        self.total_transaction_data = None

        self.current_transaction_row = None
        self.total_transaction_columns = None
        self.total_order_columns = None

        # total numpy object
        self.total_data = None
        self.total_data_columns = None

        # 총 Data 변수
        self.total_df = None

        # Trading 관련 변수
        self.decision = None
        self.prev_position = None
        self.current_ticker = None
        self.ticker_array = None
        self.rate = 0
        self.position = None

        self.limit_low = 0
        self.limit_high = 0
        self.current_open = 0
        self.current_high = 0
        self.current_low = 0
        self.current_price = 0
        self.current_transaction_amount = 0
        self.current_amount = 0

        self.prev_available = 0
        self.prev_executed = 0
        self.prev_amount = 0
        self.prev_contract_amount = 0
        self.prev_price = 0
        self.prev_net_worth = 0

        self.update_amount = 0
        self.update_executed = 0
        self.update_available = 0
        self.update_contract_amount = 0

        self.limit_total_transaction = 0
        self.limit_total_gain = 0
        self.limit_total_amount = 0
        self.total_order_executed_time = ""

        self.ticker_total_eval = 0
        self.total_eval = 0
        self.total_contract = 0

    # --- Setting Section
    # 시뮬레이션 세팅 관련
    def setting(self, setting=None):
        if setting is None:
            self.init_setting()
            self.future_setting()
            self.save_setting()
            self.limit_order_setting()
        else:
            [self.init, self.future, self.save_per_row, self.limit] = ['y' if i == 1 else 'n' for i in setting]

    # 시뮬레이션을 저장하는 DB 를 초기화 할지 말지 결정.
    def init_setting(self):
        init_ = input("DB 초기화 여부 [y/n] : ")
        if str(init_) == 'y':
            self.init = True
        elif str(init_) == 'n':
            self.init = False
        else:
            print("[y/n] 이외의 값이 입력되었습니다. 다시 입력해 주십시오")
            self.init_setting()

    # future 거래에 대한 여부 결정
    def future_setting(self):
        check = input("Future 여부 [y/n] : ")
        if str(check) == 'y':
            self.future = True
        elif str(check) == 'n':
            self.future = False
        else:
            print("[y/n] 이외의 값이 입력되었습니다. 다시 입력해 주십시오")
            self.future_setting()

    # rows 마다 저장할 것인지 여부
    def save_setting(self):
        check = input("Save to DB per rows 여부 [y/n] : ")
        if str(check) == 'y':
            self.save_per_row = True
        elif str(check) == 'n':
            self.save_per_row = False
        else:
            print("[y/n] 이외의 값이 입력되었습니다. 다시 입력해 주십시오")
            self.save_setting()

    # limit_order_setting
    def limit_order_setting(self):
        check = input("limit 여부 [y/n] : ")
        if str(check) == 'y':
            self.limit = True
        elif str(check) == 'n':
            self.limit = False
        else:
            print("[y/n] 이외의 값이 입력되었습니다. 다시 입력해 주십시오")
            self.limit_order_setting()

    # --- Simulation 을 위한 Data Getter Section
    # ticker_list 에서 ticker 를 뽑고, 지정된 ticker 에 대해 simulation time 을 setting 한 다음, DB 에서 data 를 가져옴
    def init_total_transaction_data(self, start, finish, *input_ticker):
        # DB 에는 timestamp 형식으로 저장되어 있음.

        self.simulator_setting(start, finish, *input_ticker)

        print("Get DATA From DB ...")
        self.total_df = self.get_total_df(*input_ticker)
        print("Done")

        # 시물레이터의 기본 설정 - 시작시간과 끝 시간 설정

    def simulator_setting(self, start, finish, *input_tickers):
        """
        Simulator 의 기본 세팅
        Simulator 을 시행하면서 만들어지 지는 table 의 이름 및, 시간 변수를 인스턴스 변수(str) 로 저장.
        :param start: 시뮬레이션에 사용할 데이터의 시작 시간
        :param finish: 시뮬레이션에 사용할 데이터의 최종 시간
        :return: None
        """
        self.ticker_array = np.array(input_tickers)
        ticker_name = ''.join([ticker[:-3] for ticker in input_tickers])
        total_table_name = f"{start}_{finish}_" + ticker_name
        self.total_table_name = total_table_name

        # String to datetime Object
        self.db_start_time = str(datetime.strptime(str(start), '%Y%m%d'))
        self.db_finish_time = str(datetime.strptime(str(finish), '%Y%m%d'))
        print(f"Simulation {self.db_start_time} to {self.db_finish_time}")

    def get_total_df(self, *input_tickers):

        if table_exist('simulator', self.total_table_name):
            conn = mysql_conn('simulator')
            cur = conn.cursor()
            sql = f"SELECT * FROM {self.total_table_name}"
            cur.execute(sql)
            total_df = pd.DataFrame(cur.fetchall(), columns=[i[0] for i in cur.description])
            cur.close()
            conn.close()

            return total_df
        else:
            ticker = "btckrw"
            conn = mysql_conn('coin_min_db')
            cur = conn.cursor()
            sql = f"SELECT * FROM `coin_min_db`.{ticker} WHERE " \
                  f"`coin_min_db`.{ticker}.time BETWEEN '{self.db_start_time}' AND '{self.db_finish_time}'"
            cur.execute(sql)
            columns = ['time', f'{ticker}_open', f'{ticker}_low', f'{ticker}_high', f'{ticker}_close',
                       f'{ticker}_volume']
            total_df = pd.DataFrame(cur.fetchall(), columns=columns)

            for ticker in input_tickers:
                if ticker == "btckrw":
                    continue
                sql = f"SELECT * FROM `coin_min_db`.{ticker} WHERE " \
                      f"`coin_min_db`.{ticker}.time BETWEEN '{self.db_start_time}' AND '{self.db_finish_time}'"
                cur.execute(sql)
                columns = ['time', f'{ticker}_open', f'{ticker}_low', f'{ticker}_high', f'{ticker}_close',
                           f'{ticker}_volume']
                temp = pd.DataFrame(cur.fetchall(), columns=columns)
                total_df = pd.merge(total_df, temp, how='left', on='time')

            cur.close()
            conn.close()

            total_df.to_sql(self.total_table_name, sql_connect("simulator"), index=False, if_exists="append",
                            chunksize=1440)

            return total_df

    # --- Algorithm Connection Section
    # algorithm 과 연결
    def algorithm_connection(self, algo):
        # algo connection 기본 설정.

        self.algo = algo()
        self.observed_rows = self.algo.observed_rows_num
        self.buy_ratio = self.algo.execute_ratio_buy
        self.simulator_start_time = self.total_df[0:self.observed_rows].tail(1).time.item()
        # sell_ratio = algo.execute_ratio_sell

    # --- Trading Simulation Section
    # 실제 trading simulation 부분
    # 현재가 - 종가.
    # 수수료 계산 - amount 를 적게 처리. - 매수
    # 거래 후 정산금액을 적게 처리 - 매도.
    # @time_checker
    def simulation_trading(self):

        self.init_simulation()

        # self.current_time 을 가지는 simulation row 를 DB 에 쌓아가며 simulation 을 진행.
        for i in range(len(self.total_data) - self.observed_rows):
            # initializing 이 index 0 을 기준으로 start time 을 지정에 유의
            idx = i + 1
            current_tickers_data = self.total_data[idx:idx + self.observed_rows]
            # 시간 설정 및 이전 열 데이터 Check
            if self.skip_check(current_tickers_data):
                continue

            self.init_time_related_vars()

            for ticker in self.ticker_array:
                self.current_ticker = ticker
                self.init_ticker_related_vars()
                self.init_trading_related_vars(ticker, ticker_data=self.init_ticker_data(ticker, current_tickers_data))

                # 거래를 시작하기 전, limit order 에 대해 계산할것을 고려
                # 업데이트가 필요한 변수들 : prev_amount, prev_contract_amount, current_available

                self.limit_order_trading(ticker)
                self.decision_implement()
                self.liquidation_check()
                self.update_transaction_db(ticker)

            self.update_net_worth()
            self.saving_row()

        if not self.save_per_row:
            self.total_transaction_db = pd.concat(self.total_transaction_data)
            self.total_transaction_db.to_sql(self.transaction_table_name,
                                             sql_connect('simulator'), index=False, if_exists='replace')

            pd.DataFrame(self.total_order_data, columns=self.total_order_columns).to_sql(
                self.order_list_name, sql_connect('simulator'), index=False, if_exists='replace', method='multi')

    # --- Reset Simulation Section

    def init_simulation(self):
        print("Setting Simulator DB ...")
        self.simulator_table_setting()
        print("Done")
        print(f"Tickers : {self.ticker_array}")

        print("Initializing DB ... Get Prev Data ...")
        self.sync_prev_data()
        print("Done")

        self.total_data_columns = self.total_df.columns.to_numpy(dtype=str)
        self.total_transaction_columns = self.total_transaction_db.columns.to_numpy(dtype=str)
        self.total_order_columns = self.total_order_db.columns.to_numpy(dtype=str)

        self.total_data = self.total_df.to_numpy()

    # Simulation 을 위한 DB 의 이름 설정 및 DB 생성
    def simulator_table_setting(self):
        if self.future:
            self.transaction_table_name = self.total_table_name + "_simulation" + "_future"
            self.order_list_name = self.total_table_name + "_order_list" + "_future"
        else:
            self.transaction_table_name = self.total_table_name + "_simulation"
            self.order_list_name = self.total_table_name + "_order_list"

        if self.limit:
            self.transaction_table_name += "_limit"
            self.order_list_name += ""

        if self.save_per_row:
            self.transaction_table_name += "_saved"
            self.order_list_name += "_saved"

        if not table_exist("simulator", self.transaction_table_name) or self.init is True:
            self.make_simulation_db(self.transaction_table_name, self.ticker_array, self.simulator_start_time)

        if not table_exist("simulator", self.order_list_name) or self.init is True:
            self.make_limit_order_db()

    # simulation 하는 내용을 저장할 DB 를 구성 및 초기화.
    def make_simulation_db(self, table_name, tickers, init_time):
        db_name = 'simulator'
        simulation_db = defaultdict(list)
        simulation_db["time"] = [init_time]
        simulation_db["available"] = [float(self.init_balance)]
        simulation_db["leverage"] = [float(self.leverage)]

        for ticker in tickers:
            simulation_db[f"current_{ticker}_amount"] = [0.0]
            simulation_db[f"current_{ticker}_price"] = [0.0]
            simulation_db[f"executed_{ticker}_price"] = [0.0]
            simulation_db[f"{ticker}_decision"] = ["stay"]
            simulation_db[f"{ticker}_position"] = ["close"]
            simulation_db[f"{ticker}_contract_amount"] = [0.0]
            simulation_db[f"{ticker}_rate"] = [0.0]

        simulation_db["net_worth"] = [float(self.init_balance)]

        simulation_db = pd.DataFrame.from_dict(dict(simulation_db))
        simulation_db.to_sql(table_name, sql_connect(db_name), index=False, if_exists='replace', method='multi')

        self.total_transaction_db = simulation_db.copy()

    # order list 를 저장할 DB 를 구성 및 초기화.
    def make_limit_order_db(self):
        db_name = 'simulator'
        table_name = self.order_list_name
        order_db = defaultdict(list)
        order_db['ticker'] = ['None']
        order_db['amount'] = [0.0]
        order_db['limit_order_price_high'] = [0.0]
        order_db['limit_order_price_low'] = [0.0]
        order_db['order_executed_time'] = [datetime.strptime("2011-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")]

        order_db = pd.DataFrame.from_dict(dict(order_db))
        order_db.to_sql(table_name, sql_connect(db_name), index=False, if_exists='replace', method='multi')

        self.total_order_db = order_db.copy()

    # 이전에 저장된 데이터가 있을 경우, Simulation 에 사용할 Total DB 를 이전것과 동기화 및 최종 simulation 진행 시간 업데이트
    def sync_prev_data(self):
        conn = mysql_conn('simulator')
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {self.transaction_table_name}")
        self.total_transaction_db = pd.DataFrame(cur.fetchall(), columns=[i[0] for i in cur.description])
        self.total_transaction_data = []
        self.total_transaction_data.append(self.total_transaction_db)
        self.simulated_time = self.total_transaction_db.to_numpy()[-1][0]

        cur.execute(f"SELECT * FROM {self.order_list_name}")
        self.total_order_db = pd.DataFrame(cur.fetchall(), columns=[i[0] for i in cur.description])
        self.total_order_data = self.total_order_db.to_numpy().tolist()
        # self.total_order_data = self.total_order_db.to_numpy()
        cur.close()
        conn.close()

    # 이전에 저장된 데이터를 기준으로 Skip 여부 체크
    def skip_check(self, current_tickers_data):
        # self.current_time = current_tickers_data.tail(1).time.item()
        self.current_time = current_tickers_data[-1][0]
        # simulated_time = self.total_transaction_db.tail(1).time.item()

        if str(self.simulated_time) >= str(self.current_time):
            return True
        else:
            return False

    # time 에 따라 초기화 해주어야 할 변수 및 1-row DF 초기화
    def init_time_related_vars(self):
        # 거래에 따른 총 손익 관련 변수 기본 Setting (모든 ticker 에 대한 total 값)
        self.total_eval = 0
        self.total_contract = 0

        # 이번 시간 열에 대해 새로 업데이트할 1 row 데이터 프레임 생성
        # self.current_transaction_row = self.total_transaction_db.tail(1).reset_index(drop=True).copy()
        self.current_transaction_row = self.total_transaction_data[-1].reset_index(drop=True).copy()

    # ticker 에 따라 초기화 해주어야 할 변수 초기화
    def init_ticker_related_vars(self):
        # self.leverage
        # self.self.fee
        self.sell_ratio = 1

        # _amount : 보유 량
        # self.transaction_amount : 거래 자금
        # _available : 거래 가능 자금
        # _total_eval : 실 보유 량 - 자금 단위
        # _executed : 거래 중인 평균 단가

        # 이전 시간행에서 받아오는 변수들
        # 보유량, 가격, 평균 단가, 포지션
        self.prev_amount = 0
        self.prev_price = 0
        self.prev_executed = 0
        self.prev_position = None
        self.rate = 0

        # 현재 시간행에서 거래로 인해 발생하는 변수들
        # 현재 거래 시도량, 현재 주문량, 현재가, 현재 주문 가능 자산
        self.current_transaction_amount = 0
        self.current_amount = 0
        self.current_price = 0
        self.prev_available = 0

        # 최종적으로 현재 시간행에 업데이트해줄 변수들
        # 최종 보유량, 최종 평균 단가, 최종 주문 가능 자산
        self.update_amount = 0
        self.update_executed = 0
        self.update_available = 0

        # future - 계약을 통한 거래가 아닌 경우, contract 관련 변수 제어
        self.update_contract_amount = 0
        self.prev_contract_amount = 0

        self.init_limit_order_vars()

    # limit order 관련 변수 초기화
    def init_limit_order_vars(self):
        self.limit_total_transaction = 0
        self.limit_total_gain = 0
        self.limit_total_amount = 0
        self.total_order_executed_time = ""

    # trading 에 직접적으로 연관된 변수들을 Data 를 참고하여 초기화.
    def init_trading_related_vars(self, ticker, ticker_data):
        # observation rows 에 맞는 ticker_df 가 들어가는 경우, 그에 맞는 self.decision 을 반환.
        print(f"{self.current_time} {ticker} Trading ")

        self.decision = self.algo.decision(ticker_data)['decision']
        self.limit_low = self.algo.decision(ticker_data)['limit_low']
        self.limit_high = self.algo.decision(ticker_data)['limit_high']

        # current ticker 에 대한 정보 - current_price 를 close 로 설정.
        # self.current_open = float(ticker_df.tail(1)['open'].item())
        # self.current_high = float(ticker_df.tail(1)['high'].item())
        # self.current_low = float(ticker_df.tail(1)['low'].item())
        # self.current_price = float(ticker_df.tail(1)['close'].item())
        self.current_open = float(ticker_data[-1, 0])
        self.current_high = float(ticker_data[-1, 1])
        self.current_low = float(ticker_data[-1, 2])
        self.current_price = float(ticker_data[-1, 3])

        # 바로 이전 시간 행에서 ticker 관련 변수들
        self.prev_amount = float(self.current_transaction_row[f"current_{ticker}_amount"].item())
        self.prev_price = float(self.current_transaction_row[f"current_{ticker}_price"].item())
        self.prev_executed = float(self.current_transaction_row[f"executed_{ticker}_price"].item())
        self.prev_position = str(self.current_transaction_row[f"{ticker}_position"].item())
        self.prev_available = float(self.current_transaction_row["available"].item())
        self.prev_net_worth = float(self.current_transaction_row["net_worth"].item())

        # future 설정이 존재할 경우 추가적 변수
        self.prev_contract_amount = float(self.current_transaction_row[f"{ticker}_contract_amount"].item())

    # 기존 데이터의 columns 를 일반화
    @staticmethod
    def init_ticker_df(ticker, data):
        columns = ['time', f'{ticker}_open', f'{ticker}_low', f'{ticker}_high', f'{ticker}_close',
                   f'{ticker}_volume']
        ticker_df = data[columns]

        change_dict = {f'{ticker}_open': 'open',
                       f'{ticker}_low': 'low',
                       f'{ticker}_high': 'high',
                       f'{ticker}_close': 'close',
                       f'{ticker}_volume': 'volume'}

        ticker_df = ticker_df.rename(columns=change_dict)
        return ticker_df

    def init_ticker_data(self, ticker, data):
        idx = np.where(self.total_data_columns == f"{ticker}_open")[0].item()
        ticker_data = data[:, idx:idx + 5]
        return ticker_data

    # Decision 에 따른 거래를 시행하기전, limit order list 를 첨고하여 limit order process 진행
    def limit_order_trading(self, ticker):
        if self.limit:
            # self.total_order_db.reset_index(drop=True, inplace=True)
            self.limit_order_db_check(ticker, self.prev_executed)
            if self.limit_total_gain != 0.0:
                print(f"Limit Order at {self.total_order_executed_time} be Signed")
                print(ticker, self.prev_amount, self.limit_total_amount)
                self.prev_amount -= self.limit_total_amount
                if self.prev_amount == 0:
                    self.prev_executed = 0
                if self.future:
                    update_contract_amount = np.abs(self.prev_amount / self.leverage)
                    current_realized_contract \
                        = (self.prev_contract_amount - update_contract_amount) * self.current_price
                    current_realized_contract = np.abs(current_realized_contract) * (1 - self.fee)
                    self.prev_contract_amount = update_contract_amount
                    self.limit_total_gain = np.abs(self.limit_total_gain) * (1 - self.fee) * np.sign(
                        self.limit_total_gain)
                    total_realized_gain = self.limit_total_gain + current_realized_contract
                    print(self.limit_total_gain, current_realized_contract)
                    print(total_realized_gain)
                    self.prev_available += total_realized_gain
                else:
                    self.prev_available += np.abs(self.limit_total_transaction) * (1 - self.fee) * np.sign(
                        self.limit_total_transaction)

    # limit order price 가 될 경우, 얻어지는 gain 를 계산
    def limit_order_db_check(self, ticker, executed_price):
        remove_rows = []
        for i in range(len(self.total_order_data)):
            order_list_ticker = self.total_order_data[i][0]
            if str(order_list_ticker) != str(ticker):
                continue
            limit_order_amount = self.total_order_data[i][1]
            limit_order_high = self.total_order_data[i][2]
            limit_order_low = self.total_order_data[i][3]
            gap_high = np.abs(self.current_open - limit_order_high)
            gap_low = np.abs(self.current_open - limit_order_low)
            limit_ordered_time = str(self.total_order_data[i][4])

            if limit_order_high < self.current_high and gap_high < gap_low:
                remove_rows.append(self.total_order_data[i])
                self.limit_total_gain += (limit_order_high - executed_price) * limit_order_amount
                self.limit_total_transaction += limit_order_high * limit_order_amount
                self.limit_total_amount += limit_order_amount
                self.total_order_executed_time += limit_ordered_time + " "

            elif limit_order_low > self.current_low and gap_low < gap_high:
                remove_rows.append(self.total_order_data[i])
                self.limit_total_gain += (limit_order_low - executed_price) * limit_order_amount
                self.limit_total_transaction += limit_order_low * limit_order_amount
                self.limit_total_amount += limit_order_amount
                self.total_order_executed_time += limit_ordered_time + " "

        for row in remove_rows:
            self.total_order_data.remove(row)

    # Decision 에 대한 시행:
    def decision_implement(self):
        if self.decision == 'buy':
            self.decision_buy()

        elif self.decision == 'sell':
            self.decision_sell()

        if self.decision == 'close':
            self.decision_close()

        if self.decision == 'stay':
            self.decision_stay()

    # limit order 의 형성
    def limit_order_add(self, ticker, amount, limit_high, limit_low, executed_time):
        if amount == 0:
            return None

        order_list = [ticker, amount, float(limit_high), float(limit_low), executed_time]
        self.total_order_data.append(order_list)

    def decision_buy(self):
        if not self.future:
            self.current_transaction_amount = round(self.buy_ratio * self.init_balance, 0)
            if self.prev_available < self.current_transaction_amount:
                self.decision = 'stay'
            else:
                self.current_amount = (self.current_transaction_amount / self.current_price) * (1 - self.fee)
                self.update_amount = self.prev_amount + self.current_amount
                self.update_available = self.prev_available - self.current_transaction_amount
                self.ticker_total_eval = (
                        self.prev_amount * self.prev_executed + self.current_amount * self.current_price)
                self.update_executed = self.ticker_total_eval / self.update_amount
                self.position = "long"
                if self.update_executed != 0:
                    self.rate = (self.current_price - self.update_executed) / self.update_executed * 100

        # Future
        else:
            contract_amount = round(self.buy_ratio * self.init_balance, 0)
            coin_contract_amount = contract_amount / self.current_price * (1 - self.fee)
            self.current_amount = coin_contract_amount * self.leverage
            # 매수 : amount 증가 . self.current_amount > 0

            # 공매도 표지션 부분/전체 닫기 - 공매도 포지션 : prev_amount < 0
            if self.prev_amount < 0:
                # 현재 추가할 포지션 양이 이전 양 보다 작을 경우, 공매도 포지션 유지.
                if np.abs(self.prev_amount) >= np.abs(self.current_amount):
                    # 공매도 포지션을 일부 닫으므로, 지금 양 만큼 이전 계약금 소거 및 남은 계약 만큼 공매도 포지션 유지.
                    self.update_contract_amount = self.prev_contract_amount - coin_contract_amount
                    self.update_amount = - self.update_contract_amount * self.leverage
                    self.update_executed = self.prev_executed
                    # 공매도 포지션 일부 종료에 따른 손익. self.current_amount ( > 0 )만큼 포지션이 종료
                    realized_gain = - self.current_amount * (self.current_price - self.prev_executed)
                    realized_gain = np.abs(realized_gain) * (1 - self.fee) * np.sign(realized_gain)
                    # 지금 양 만큼 이전 계약의 회수
                    realized_contract = coin_contract_amount * self.current_price
                    realized_contract = np.abs(realized_contract) * (1 - self.fee) * np.sign(realized_contract)
                    # 닫아진 계약만큼의 손익과 계약금 회수
                    self.update_available = self.prev_available + realized_gain + realized_contract

                # 공매도 계약에 대한 포지션을 모두 닫고, 일부 남은 공매수 포지션을 열기.
                else:
                    # 공매도 포지션을 닫고, 이전 계약금 소거 및 새로운 계약 만큼 공매수 포지션 생성
                    self.update_contract_amount = np.abs(self.prev_contract_amount - coin_contract_amount)
                    self.update_amount = self.update_contract_amount * self.leverage
                    self.update_executed = self.current_price
                    # 공매도 포지션 종료에 따른 손익. self.prev_amount ( < 0) 만큼 포지션이 종료
                    realized_gain = self.prev_amount * (self.current_price - self.prev_executed)
                    realized_gain = np.abs(realized_gain) * (1 - self.fee) * np.sign(realized_gain)
                    # 이전 포지션 계약의 전량 회수
                    realized_contract = (self.prev_contract_amount - self.update_contract_amount) * self.current_price
                    realized_contract = np.abs(realized_contract) * (1 - self.fee) * np.sign(realized_contract)
                    # 닫아진 계약만큼의 손익과 계약금 회수
                    self.update_available = self.prev_available + realized_gain + realized_contract

                # 포지션 전체 종료 경우
                if self.update_amount == 0:
                    self.update_executed = 0
                    self.rate = 0
                    self.position = 'close'
                # update amount 에 따라 최종 포지션 결정.
                elif self.update_amount < 0:
                    self.position = 'short'
                    self.rate = - (self.current_price - self.update_executed) / self.update_executed * 100
                elif self.update_amount > 0:
                    self.position = 'long'
                    self.rate = (self.current_price - self.update_executed) / self.update_executed * 100

            # 공매수 포지션 추가
            else:
                self.update_amount = self.prev_amount + self.current_amount
                self.update_available = self.prev_available - contract_amount
                self.update_contract_amount = self.prev_contract_amount + coin_contract_amount

                if self.update_available < 0:
                    self.decision = 'stay'
                else:
                    self.ticker_total_eval = (
                            self.prev_amount * self.prev_executed + self.current_amount * self.current_price)
                    self.update_executed = self.ticker_total_eval / self.update_amount
                    self.position = 'long'
                    self.rate = (self.current_price - self.update_executed) / self.update_executed * 100

        if self.limit and self.decision == 'buy':
            self.limit_order_add(self.current_ticker, self.current_amount,
                                 self.limit_high, self.limit_low, self.current_time)

    def decision_sell(self):
        # 전량매도.
        if not self.future:
            if self.prev_amount == 0:
                self.decision = 'stay'
            elif self.sell_ratio != 1:
                self.current_transaction_amount = self.prev_amount * self.sell_ratio * self.prev_executed
                sold_amount = self.current_transaction_amount * (1 - self.fee)
                self.update_available = self.prev_available + sold_amount
                self.update_executed = self.prev_executed
                self.update_amount = self.prev_amount * (1 - self.sell_ratio)
                self.position = "long"
                self.rate = (self.current_price - self.update_executed) / self.update_executed * 100
            else:
                self.current_transaction_amount = self.prev_amount * self.prev_executed
                sold_amount = self.current_transaction_amount * (1 - self.fee)
                self.update_available = self.prev_available + sold_amount
                self.update_executed = 0
                self.update_amount = 0
                self.position = "close"
                self.rate = 0

                # self.total_eval += 0

        # Future
        else:
            contract_amount = round(self.buy_ratio * self.init_balance, 0)
            coin_contract_amount = contract_amount / self.current_price * (1 - self.fee)
            self.current_amount = - coin_contract_amount * self.leverage
            # 매도 : amount 감소. self.current_amount < 0

            # 공매수 포지션 일부/전체 닫기. - 공매수 포지션 : prev_amount > 0
            if self.prev_amount > 0:
                # 현재 추가할 포지션 양이 이전 양 보다 작을 경우, 공매수 포지션 유지.
                if np.abs(self.prev_amount) >= np.abs(self.current_amount):
                    # 공매수 포지션을 일부 닫으므로, 지금 양 만큼 이전 계약금 소거
                    self.update_contract_amount = self.prev_contract_amount - coin_contract_amount
                    self.update_amount = self.update_contract_amount * self.leverage
                    self.update_executed = self.prev_executed
                    # 공매수 포지션 일부 종료에 대한 손익. self.current_amount( < 0) 만큼 포지션이 종료.
                    realized_gain = - self.current_amount * (self.current_price - self.prev_executed)
                    realized_gain = np.abs(realized_gain) * (1 - self.fee) * np.sign(realized_gain)
                    # 지금 양 만큼 이전 계약의 회수
                    realized_contract = coin_contract_amount * self.current_price
                    realized_contract = np.abs(realized_contract) * (1 - self.fee) * np.sign(realized_contract)
                    # 닫아진 계약만큼의 손익과 계약금 회수
                    self.update_available = self.prev_available + realized_gain + realized_contract

                # 공매수 계약에 대한 포지션을 모두 닫고, 일부 남은 공매도 포지션을 열기.
                else:
                    # 공매수 포지션을 닫고, 이전 계약금 소거 및 새로운 계약 만큼 공매도 포지션 생성.
                    self.update_contract_amount = np.abs(self.prev_contract_amount - coin_contract_amount)
                    self.update_amount = - self.update_contract_amount * self.leverage
                    self.update_executed = self.current_price
                    # 공매수 포지션 종료에 따른 손익. self.prev_amount( > 0 ) 만큼 포지션이 종료
                    realized_gain = self.prev_amount * (self.current_price - self.prev_executed)
                    realized_gain = np.abs(realized_gain) * (1 - self.fee) * np.sign(realized_gain)
                    # 이전 포지션 계약의 전량 회수
                    realized_contract = (self.prev_contract_amount - self.update_contract_amount) * self.current_price
                    realized_contract = np.abs(realized_contract) * (1 - self.fee) * np.sign(realized_contract)
                    # 닫아진 계약만큼의 손익과 계약금 회수
                    self.update_available = self.prev_available + realized_gain + realized_contract

                # 포지션 전체 종료 경우
                if self.update_amount == 0:
                    self.update_executed = 0
                    self.rate = 0
                    self.position = 'close'
                # update amount 에 따라 최종 포지션 결정
                elif self.update_amount < 0:
                    self.position = 'short'
                    self.rate = - (self.current_price - self.update_executed) / self.update_executed * 100
                elif self.update_amount > 0:
                    self.position = 'long'
                    self.rate = (self.current_price - self.update_executed) / self.update_executed * 100

            # 공매도 포지션 추가.
            else:
                self.update_amount = self.prev_amount + self.current_amount
                self.update_available = self.prev_available - contract_amount
                self.update_contract_amount = self.prev_contract_amount + coin_contract_amount

                if self.update_available < 0:
                    self.decision = 'stay'
                else:
                    self.ticker_total_eval = (
                            self.prev_amount * self.prev_executed + self.current_amount * self.current_price)
                    self.update_executed = self.ticker_total_eval / self.update_amount
                    self.position = 'short'
                    self.rate = -(self.current_price - self.update_executed) / self.update_executed * 100

        if self.limit and self.decision == 'sell':
            self.limit_order_add(self.current_ticker, self.current_amount,
                                 self.limit_high, self.limit_low, self.current_time)

    def decision_stay(self):
        self.update_available = self.prev_available
        self.update_amount = self.prev_amount
        self.update_executed = self.prev_executed

        if not self.future:
            if self.update_executed != 0:
                self.position = "long"
                self.rate = (self.current_price - self.update_executed) / self.update_executed * 100
            else:
                self.position = "close"
                self.rate = 0
        else:
            self.update_contract_amount = self.prev_contract_amount
            if self.prev_amount == 0:
                self.position = 'close'
            else:
                if self.prev_position == "long":
                    self.rate = (self.current_price - self.update_executed) / self.update_executed * 100
                    self.position = 'long'
                elif self.prev_position == "short":
                    self.rate = - (self.current_price - self.update_executed) / self.update_executed * 100
                    self.position = 'short'
                else:
                    self.position = 'close'
                    self.rate = 0

    def decision_close(self):
        if self.future:
            self.update_amount = 0
            self.update_contract_amount = 0
            self.update_executed = 0
            self.rate = 0
            # self.prev_amount < 0 then self.current_price 가 낮아야 이득.
            # self.prev_amount > 0 then self.current_price 가 높아야 이득.
            realized_gain = self.prev_amount * (self.current_price - self.prev_executed)
            realized_gain = np.abs(realized_gain) * (1 - self.fee) * np.sign(realized_gain)
            # 이전 계약 전체회수
            realized_contract = self.prev_contract_amount * self.current_price
            realized_contract = np.abs(realized_contract) * (1 - self.fee) * np.sign(realized_contract)
            self.update_available = self.prev_available + realized_gain + realized_contract
            self.position = 'close'
        else:
            self.decision = 'stay'

    def liquidation_check(self):
        if self.future:
            # self.current_price > self.update_executed : self.update_amount > 0 이면 이득
            # self.current_price < self.update_executed : self.update_amount < 0 이면 이득

            if self.rate < 0 and self.update_contract_amount < np.abs(self.update_amount * (self.rate / 100)):
                print(f"{self.current_ticker} 청산.")
                self.update_contract_amount = 0
                self.position = 'close'
                self.rate = 0

    def update_transaction_db(self, ticker):
        update_db = defaultdict(list)
        update_db["time"] = [self.current_time]
        update_db["available"] = [float(self.update_available)]
        update_db["leverage"] = [float(self.leverage)]
        update_db[f"current_{ticker}_amount"] = [self.update_amount]
        update_db[f"current_{ticker}_price"] = [self.current_price]
        update_db[f"executed_{ticker}_price"] = [self.update_executed]
        update_db[f"{ticker}_decision"] = [self.decision]
        update_db[f"{ticker}_position"] = [self.position]
        update_db[f"{ticker}_rate"] = [self.rate]
        update_db[f"{ticker}_contract_amount"] = [self.update_contract_amount]
        update_db = pd.DataFrame.from_dict(dict(update_db))

        if not self.future:
            # 총 평가 금액
            self.total_eval += np.abs(self.update_amount * self.current_price)
        else:
            self.total_contract += self.update_contract_amount * self.current_price

        self.current_transaction_row.update(update_db)

    # net_worth update
    def update_net_worth(self):
        update_net_worth = self.update_available + self.total_eval \
            if not self.future else self.update_available + self.total_contract
        update_net_worth_db = pd.DataFrame([update_net_worth], columns=['net_worth'])
        self.current_transaction_row.update(update_net_worth_db)

    # 모든 ticker 에 대해 만들어진 row 를 저장 - save_per_row 에 따라 다른 시행.
    def saving_row(self):
        # self.total_transaction_db = self.total_transaction_db.append(self.current_transaction_row, ignore_index=True)
        self.total_transaction_data.append(self.current_transaction_row)
        if self.save_per_row:
            self.current_transaction_row.to_sql(self.transaction_table_name,
                                                sql_connect('simulator'), index=False, if_exists='append')
            pd.DataFrame(self.total_order_data, columns=self.total_order_columns).to_sql(
                self.order_list_name, sql_connect('simulator'), index=False, if_exists='replace', method='multi')


# ---


def render_results(transaction_table_name, total_table_name, *tickers):
    conn = mysql_conn('simulator')
    cur = conn.cursor()
    sql = f"SELECT * FROM {transaction_table_name}"
    cur.execute(sql)
    total_results = pd.DataFrame(cur.fetchall(), columns=[i[0] for i in cur.description])
    total_results = total_results[1:]

    sql = f"SELECT * FROM {total_table_name}"
    cur.execute(sql)
    total_data = pd.DataFrame(cur.fetchall(), columns=[i[0] for i in cur.description])
    total_data = total_data[1:]

    for ticker in tickers:
        print(f"Rendering {ticker} simulation results")
        # render_ticker_results(ticker, total_results, total_data)
        render_ticker_results_plotly(ticker, total_results, total_data)

    print("Done")


def render_ticker_results(ticker, total_results, total_data):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1,
                                             gridspec_kw={'hspace': 0.1, 'height_ratios': [2, 1, 1, 2]},
                                             figsize=(13, 10))

    ax1.plot(total_results[f"current_{ticker}_price"], color="k", alpha=0.5, label=f"{ticker[:-3]}_price")

    x = np.argwhere(total_results[f"current_{ticker}_amount"].diff().fillna(0).to_numpy() > 0)
    y = total_results[f"current_{ticker}_price"].to_numpy()[x]
    ax1.scatter(x, y, s=5, color='r', alpha=0.5)

    x = np.argwhere(total_results[f"current_{ticker}_amount"].diff().fillna(0).to_numpy() < 0)
    y = total_results[f"current_{ticker}_price"].to_numpy()[x]
    ax1.scatter(x, y, s=5, color='b', alpha=0.5)

    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.legend(loc='upper left', fontsize=7)

    from matplotlib.patches import Patch

    colors = ['r' if i > 0 else 'b' for i in total_data[f"{ticker}_close"] - total_data[f"{ticker}_open"]]
    legend = ax2.legend(handles=[Patch(facecolor='red'), Patch(facecolor='blue')], labels=['', 'volume'], ncol=2,
                        handlelength=1, columnspacing=-0.75, fontsize=7)

    ax2.bar(np.arange(len(total_data)), total_data[f"{ticker}_volume"].to_numpy(), color=colors)

    ax2.add_artist(legend)
    ax2.set_xticklabels([])
    ax2.set_xticks([])

    ax3.plot(total_results[f"current_{ticker}_amount"].to_numpy(), color="k", alpha=0.6, label=f"{ticker} amount")
    ax3.set_xticklabels([])
    ax3.set_xticks([])
    ax3.legend(loc='upper left', fontsize=7)

    ax4.plot(total_results.net_worth.to_numpy(), color="k", alpha=0.8, label='net worth')
    ax4.set_xticks([1, len(total_results)])
    ax4.set_xticklabels([str(total_results.time.to_numpy()[1]), str(total_results.time.to_numpy()[-1])])
    ax4.legend(loc='upper left', fontsize=7)

    plt.show()


def render_ticker_results_plotly(ticker, total_results, total_data):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    total_data["interval_max"] = total_data["high"][::-1].rolling(60).max()[::-1]

    fig = make_subplots(rows=4, cols=1,
                        vertical_spacing=0.2, row_heights=[0.5, 0.1, 0.1, 0.2])

    fig.add_trace(go.Scatter(x=total_results["time"], y=total_results[f"current_{ticker}_price"],
                             mode='lines', marker=dict(color="black"), opacity=0.5,
                             name=f"{ticker[:-3]}_price"), row=1, col=1)

    fig.add_trace(go.Scatter(x=total_results["time"], y=total_results["interval_max"],
                             mode='lines', marker=dict(color="blue"), opacity=0.5,
                             name=f"interval max"), row=1, col=1)

    idx = np.argwhere(total_results[f"current_{ticker}_amount"].diff().fillna(0).to_numpy() > 0).flatten()
    x = total_results["time"].iloc[idx]
    y = total_results[f"current_{ticker}_price"].iloc[idx]
    fig.add_trace(go.Scatter(x=x, y=y,
                             mode='markers', marker=dict(size=12), fillcolor='red', opacity=0.5,
                             name=f"Buy"), row=1, col=1)

    idx = np.argwhere(total_results[f"current_{ticker}_amount"].diff().fillna(0).to_numpy() < 0).flatten()
    x = total_results["time"].iloc[idx]
    y = total_results[f"current_{ticker}_price"].iloc[idx]

    fig.add_trace(go.Scatter(x=x, y=y,
                             mode='markers', marker=dict(size=12), fillcolor='blue', opacity=0.5,
                             name=f"Sell"), row=1, col=1)

    colors = ['red' if i > 0 else 'blue' for i in total_data[f"{ticker}_close"] - total_data[f"{ticker}_open"]]

    fig.add_trace(go.Bar(x=total_data["time"], y=total_data[f"{ticker}_volume"],
                         marker=dict(color=colors)), row=2, col=1)

    fig.add_trace(go.Scatter(x=total_results["time"], y=total_results[f"current_{ticker}_amount"],
                             mode='lines', marker=dict(color='black'), opacity=0.6,
                             name=f"{ticker} amount"), row=3, col=1)

    fig.add_trace(go.Scatter(x=total_results["time"], y=total_results.net_worth,
                             mode='lines', marker=dict(color='black'), opacity=0.8,
                             name='net worth'), row=4, col=1)

    fig.update_layout(xaxis1_rangeslider_visible=True, xaxis1_rangeslider_thickness=0.07)
    fig.update_layout(xaxis2_rangeslider_visible=True, xaxis2_rangeslider_thickness=0.07)
    fig.update_layout(xaxis3_rangeslider_visible=True, xaxis2_rangeslider_thickness=0.07)
    fig.update_layout(xaxis4_rangeslider_visible=True, xaxis2_rangeslider_thickness=0.07)


    fig.show()
    fig.write_html(f"./simulation_results_{ticker}.html")


if __name__ == "__main__":

    if input(" Mode : 1. Simulation / 2. Rendering") == "1":
        start_time = time.time()
        s = Simulator(1000000, 0)
        s.setting()
        s.leverage = 10
        s.init_total_transaction_data(20201020, 20201023, "btckrw")
        s.algorithm_connection(algo=LGBMAlgorithm)
        s.simulation_trading()
        print(f"time {time.time() - start_time}")

        start_time = time.time()
        render_results(s.transaction_table_name, s.total_table_name,"btckrw")
        print(f"time {time.time() - start_time}")

    else:
        start_time = time.time()
        render_results('20210407_20210408_btceth_simulation_limit', '20210407_20210408_btceth', "btckrw", "ethkrw")
        print(f"time {time.time() - start_time}")

    # 1440분 10 종목 시뮬 시간 ~ 337s
    # none save rows, 2 종목 28.
