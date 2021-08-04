from mysql_func import *
from min_algorithm import *
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
# 안정성 테스트
# (Future, DB init, Save, Limit)
# y,y,y,y == y,y,n,y

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
        self.save_ = True
        self.limit = False
        self.liquidation = False
        self.fee = 0.25/100


    # 시뮬레이션을 저장하는 DB 를 초기화 할지 말지 결정.
    def init_simulation_whether(self):
        init_ = input("DB 초기화 여부 [y/n] : ")
        if str(init_) == 'y':
            self.init = True
        elif str(init_) == 'n':
            self.init = False
        else:
            print("[y/n] 이외의 값이 입력되었습니다. 다시 입력해 주십시오")
            self.init_simulation_whether()

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
        if self.init:
            check = input("Save to DB per rows 여부 [y/n] : ")
            if str(check) == 'y':
                self.save_ = True
            elif str(check) == 'n':
                self.save_ = False
            else:
                print("[y/n] 이외의 값이 입력되었습니다. 다시 입력해 주십시오")
                self.save_setting()
        else:
            pass
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

    # simulation 하는 내용을 저장할 DB 를 구성 및 초기화.
    def make_simulation_db(self, table_name, ticker_list, init_time):
        db_name = 'simulator'
        simulation_db = defaultdict(list)
        simulation_db["time"] = [init_time]
        simulation_db["available"] = [float(self.init_balance)]
        simulation_db["leverage"] = [float(self.leverage)]

        for ticker in ticker_list:
            simulation_db[f"current_{ticker}_amount"] = [0.0]
            simulation_db[f"current_{ticker}_price"] = [0.0]
            simulation_db[f"executed_{ticker}_price"] = [0.0]
            simulation_db[f"{ticker}_decision"] = ["stay"]
            simulation_db[f"{ticker}_position"] = ["close"]
            simulation_db[f"{ticker}_contract_amount"] = [0.0]
            simulation_db[f"{ticker}_rate"] = [0.0]

        simulation_db["net_worth"] = [float(self.init_balance)]

        simulation_db = pd.DataFrame.from_dict(dict(simulation_db))
        self.init_simulation_df = simulation_db
        self.simulation_columns = simulation_db.columns
        simulation_db.to_sql(table_name, sql_connect(db_name), index=False, if_exists='replace', method='multi')

    def make_limit_order_db(self):
        db_name = 'simulator'
        table_name = self.simulation_order_list
        order_list = defaultdict(list)
        order_list['ticker'] = ['None']
        order_list['amount'] = [0.0]
        order_list['limit_order_price_high'] = [0.0]
        order_list['limit_order_price_low'] = [0.0]
        order_list['order_executed_time'] = [datetime.strptime("2011-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")]

        order_list = pd.DataFrame.from_dict(dict(order_list))
        order_list.to_sql(table_name, sql_connect(db_name), index=False, if_exists='replace', method='multi')
        if not self.save_:
            self.init_order_list_df = order_list

    def get_limit_order_db(self):
        conn = mysql_conn('simulator')
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {self.simulation_order_list}")
        columns = [i[0] for i in cur.description]
        self.total_order_list = pd.DataFrame(cur.fetchall(), columns=columns).reset_index(drop=True)
        cur.close()
        conn.close()

    def limit_order_add(self, ticker, amount, executed_time, limit_low, limit_high):
        if amount == 0:
            return None
        db_name = 'simulator'
        table_name = self.simulation_order_list
        order_list = defaultdict(list)
        order_list['ticker'] = [ticker]
        order_list['amount'] = [amount]
        order_list['limit_order_price_high'] = [float(limit_high)]
        order_list['limit_order_price_low'] = [float(limit_low)]
        order_list['order_executed_time'] = [executed_time]

        order_list = pd.DataFrame.from_dict(dict(order_list))
        if self.save_:
            last_index = int(self.total_order_list.index[-1])
            order_list.to_sql(table_name, sql_connect(db_name), index=last_index + 1, if_exists='append', method='multi')
        else:
            self.total_limit_order_df_for_db.append(order_list, ignore_index=True)
            self.total_limit_order_df_for_db.reset_index(drop=True)


    # limit order 이 되었을 때, 음의 gain 이 발생할 경우 available 업데이트
    # 손해서보는 지정가 매도가 계속하여 발생하는 경우, availabe 이 음수로 바뀜.
    def limit_order_db_check(self, ticker, executed_price):
        if self.save_:
            pass
        else:
            self.total_order_list = self.total_limit_order_df_for_db

        for i in range(len(self.total_order_list)):
            order_list_ticker = self.total_order_list.loc[i].ticker
            if str(order_list_ticker) != str(ticker):
                continue
            limit_order_amount = self.total_order_list.loc[i].amount
            limit_order_low = self.total_order_list.loc[i].limit_order_price_low
            limit_order_high = self.total_order_list.loc[i].limit_order_price_high
            gap_high = np.abs(self.current_open - limit_order_high)
            gap_low = np.abs(self.current_open - limit_order_low)
            limit_ordered_time = str(self.total_order_list.loc[i].order_executed_time)
            if limit_order_high < self.current_high and gap_high < gap_low:
                self.total_order_list.drop(i, inplace=True)
                self.limit_total_gain += (limit_order_high - executed_price) * limit_order_amount
                self.limit_total_transaction += limit_order_high * limit_order_amount
                self.limit_total_amount += limit_order_amount
                self.total_order_executed_time += limit_ordered_time + " "
                # print(self.limit_total_gain, self.limit_total_amount)
            elif limit_order_low > self.current_low and gap_low < gap_high:
                self.total_order_list.drop(i, inplace=True)
                self.limit_total_gain += (limit_order_low - executed_price) * limit_order_amount
                self.limit_total_transaction += limit_order_low * limit_order_amount
                self.limit_total_amount += limit_order_amount
                self.total_order_executed_time += limit_ordered_time + " "
                # print(self.limit_total_gain, self.limit_total_amount)

        if self.save_:
            self.total_order_list.to_sql(self.simulation_order_list, sql_connect('simulator'), index=False,
                                         if_exists='replace', method='multi')

    # 전체 ticker list 가져오기
    def get_ticker_list(self):
        # Get ticker_List
        get_ticker_list = f"SELECT ticker FROM `setting`.ticker_list"
        cur = mysql_cursor("setting")
        cur.execute(get_ticker_list)
        ticker_list = cur.fetchall()
        ticker_list = np.array(ticker_list).flatten()
        ticker_list = np.array([ticker for ticker in ticker_list if ticker[0:3] == "KRW"])

        ticker_list_for_db = np.array([f"{ticker[4:].lower()}krw" for ticker in ticker_list])
        self.ticker_list_for_db = ticker_list_for_db

    # simulation 에 사용할 전체 DB 를 형성
    def total_df_making(self):
        total_df = pd.DataFrame()
        percentage = 1
        for ticker in self.ticker_list_for_db:
            print(f"Get {ticker} DB from database ... {percentage / len(self.ticker_list_for_db) * 100:.3f}%")
            sql = f"SELECT * FROM `coin_min_db`.{ticker} db WHERE " \
                  f"db.time BETWEEN '{self.db_start_time}' AND '{self.db_finish_time}'"
            cur = mysql_cursor("coin_min_db")
            cur.execute(sql)
            db = cur.fetchall()
            columns = ['time', f'{ticker}_open', f'{ticker}_low', f'{ticker}_high', f'{ticker}_close',
                       f'{ticker}_volume']
            db = pd.DataFrame(db, columns=columns)

            if total_df.empty:
                total_df = db
            else:
                total_df = pd.merge(total_df, db, how="left", on='time')
            percentage += 1

        total_df.to_sql(self.total_table_name, sql_connect("simulator"), index=False, if_exists="append",
                        chunksize=1440)

    # 사용할 ticker 을 설정
    def selected_input_ticker(self, input_ticker):
        conn = mysql_conn('simulator')
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {self.total_table_name}")
        columns = [i[0] for i in cur.description]  # query 결과에 따른 DATAFRAME 의 columns
        self.total_df = pd.DataFrame(cur.fetchall(), columns=columns)

        input_ticker_list = np.array(input_ticker)
        self.ticker_list_for_db = input_ticker_list
        self.columns_list \
            = [[f'{ticker}_open', f'{ticker}_low', f'{ticker}_high', f'{ticker}_close', f'{ticker}_volume']
               for ticker in input_ticker_list]

        cur.close()
        conn.close()

    # 딱히 설정하고 싶지 않은경우, 전체 ticker 중 랜덤으로 ticker 를 가져옴.
    def random_input_ticker(self, input_list):
        conn = mysql_conn('simulator')
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {self.total_table_name}")
        columns = [i[0] for i in cur.description]  # query 결과에 따른 DATAFRAME 의 columns
        self.total_df = pd.DataFrame(cur.fetchall(), columns=columns)

        random_ticker_list = np.random.choice(self.ticker_list_for_db, int(input_list[1]))
        self.ticker_list_for_db = random_ticker_list
        self.columns_list = \
            [[f'{ticker}_open', f'{ticker}_low', f'{ticker}_high', f'{ticker}_close', f'{ticker}_volume']
             for ticker in random_ticker_list]
        cur.close()
        conn.close()

    # 시간별로 저장된 DB 에서, ticker 에 맞는 모든 데이터를 변환하여 dataframe 형태로 self.total_df 에 저장
    def get_data_from_db(self, input_ticker):
        if not table_exist("simulator", self.total_table_name, sql_connect("simulator")):
            self.total_df_making()
        if type(input_ticker) == list:
            if input_ticker[0] == "random":
                self.random_input_ticker(input_ticker)
            else:
                self.selected_input_ticker(input_ticker)

            self.columns_list.insert(0, ['time'])
            self.columns_list = sum(self.columns_list, [])
            self.columns_list = np.array(self.columns_list).flatten()
            self.total_df = self.total_df[self.columns_list]

        else:
            conn = mysql_conn('simulator')
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM {self.total_table_name}")
            columns = [i[0] for i in cur.description]  # query 결과에 따른 DATAFRAME 의 columns
            self.total_df = pd.DataFrame(cur.fetchall(), columns=columns)
            cur.close()
            conn.close()

    # 시물레이터의 기본 설정 - 시작시간과 끝 시간 설정
    def simulator_time_setting(self, start, finish):
        db_check('simulator')

        total_table_name = f"{start}_{finish}"
        self.total_table_name = total_table_name

        # String to datetime Object
        self.db_start_time = str(datetime.strptime(str(start), '%Y%m%d'))
        self.db_finish_time = str(datetime.strptime(str(finish), '%Y%m%d'))
        print(f"Simulation {self.db_start_time} to {self.db_finish_time}")

    # ticker_list 에서 ticker 를 뽑고, 지정된 ticker 에 대해 simulation time 을 setting 한 다음, DB 에서 data 를 가져옴
    def get_ticker_data_from_db(self, start_time, finish_time, input_ticker=None):
        # DB 에는 timestamp 형식으로 저장되어 있음.

        self.simulator_time_setting(start_time, finish_time)

        print("Get all ticker_list ...")
        self.get_ticker_list()
        print("Done")

        print("Get DATA From DB ...")
        self.get_data_from_db(input_ticker)
        print("Done")

        return self.total_df

    # simulator 이 진행된 부분을 저장하는 DB 를 형성. 이전까지의 setting (time, name) 에 따라 DB 에 테이블 저장.
    # 만약 이미 table 이 형성되어있을 경우, columns 를 클래스 변수에 저장
    # simulator_table_setting : self.simulation_table_name & self.make_simulation_db

    # 시뮬레이션 start time 을 기준으로 1 row 의 데이터로 이루어진 initialized 된 DB 생성
    # 이미 DB가 존재한다면, pass
    def simulator_table_setting(self):
        if self.future:
            self.simulation_table_name = self.total_table_name + "_simulation" + "_future"
            self.simulation_order_list = self.total_table_name + "_order_list" + "_future"
        else:
            self.simulation_table_name = self.total_table_name + "_simulation"
            self.simulation_order_list = self.total_table_name + "_order_list"

        if self.limit:
            self.simulation_table_name += "_limit"

        if self.save_:
            self.simulation_table_name += "_saved"
            self.simulation_order_list += "_saved"


        if not table_exist("simulator", self.simulation_table_name) or self.init is True:
            self.make_simulation_db(self.simulation_table_name, self.ticker_list_for_db, self.simulator_start_time)
        else:  # 계속진행 가능하게 함수로 - resume simulation
            conn = mysql_conn('simulator')
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM {self.simulation_table_name}")
            self.simulation_columns = [i[0] for i in cur.description]
            cur.close()
            conn.close()

    # prev_time 이라는 time 에 대해, DB 에서 해당되는 time - 행을 읽어 prev_data 를 DataFrame 으로 저장
    def get_prev_row(self):
        conn = mysql_conn('simulator')
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {self.simulation_table_name} ORDER BY time DESC LIMIT 1")
        columns = [i[0] for i in cur.description]
        prev_data = pd.DataFrame(cur.fetchall(), columns=columns)
        cur.close()
        conn.close()
        return prev_data

    def setting_trader_ticker_df(self, ticker, data):
        columns = ['time', f'{ticker}_open', f'{ticker}_low', f'{ticker}_high', f'{ticker}_close',
                   f'{ticker}_volume']
        ticker_df = data[columns]

        change_dict = {f'{ticker}_open': 'open',
                       f'{ticker}_low': 'low',
                       f'{ticker}_high': 'high',
                       f'{ticker}_close': 'close',
                       f'{ticker}_volume': 'volume'}

        ticker_df = ticker_df.rename(columns=change_dict)

        # observation rows 에 맞는 ticker_df 가 들어가는 경우, 그에 맞는 self.decision 을 반환.
        self.decision = self.algo.decision(ticker_df)['decision']
        self.limit_low = self.algo.decision(ticker_df)['limit_low']
        self.limit_high = self.algo.decision(ticker_df)['limit_high']


        # 현재 시간 행에서 업데이트가 지속적으로 필요한 변수들
        self.current_available = float(self.current_total_db["available"].item())
        self.current_net_worth = float(self.current_total_db["net_worth"].item())
        self.current_price = float(ticker_df['close'].values[-1])

        # current ticker 에 대한 정보
        self.current_open = float(ticker_df['open'].values[-1])
        self.current_high = float(ticker_df['high'].values[-1])
        self.current_low = float(ticker_df['low'].values[-1])


        # 바로 이전 시간 행에서 ticker 관련 변수들
        self.prev_amount = float(self.prev_data[f"current_{ticker}_amount"].item())
        self.prev_price = float(self.prev_data[f"current_{ticker}_price"].item())
        self.prev_executed = float(self.prev_data[f"executed_{ticker}_price"].item())
        self.prev_position = str(self.prev_data[f"{ticker}_position"].item())

        # future 설정이 존재할 경우 추가적 변수
        self.prev_contract_amount = float(self.prev_data[f"{ticker}_contract_amount"].item())

    def early_stopping(self):
        if self.current_open == 0:
            return True

    def init_trader_variables_setting(self):
        # 거래에 따른 총 손익 관련 변수 기본 Setting (모든 ticker 에 대한 total 값)
        self.total_eval = 0
        self.total_contract = 0

    def init_limit_setting_ticker(self):
        self.total_order_list = None
        self.limit_total_transaction = 0
        self.limit_total_gain = 0
        self.limit_total_amount = 0
        self.total_order_executed_time = ""

    def init_trader_variables_setting_ticker(self):
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
        self.current_available = 0

        # 최종적으로 현재 시간행에 업데이트해줄 변수들
        # 최종 보유량, 최종 평균 단가, 최종 주문 가능 자산
        self.update_amount = 0
        self.update_executed = 0
        self.update_available = 0

        # future - 계약을 통한 거래가 아닌 경우, contract 관련 변수 제어
        self.update_contract_amount = 0
        self.prev_contract_amount = 0

    # algorithm 과 연결
    def algorithm_connection(self, algo=Algorithm):
        # algo connection 기본 설정.

        self.algo = algo()
        self.observed_rows = self.algo.observed_rows_num
        self.buy_ratio = self.algo.execute_ratio_buy
        self.simulator_start_time = self.total_df[0:self.observed_rows].tail(1).time.item()

        # sell_ratio = algo.execute_ratio_sell

    def setting_prev_data(self):
        if self.save_:

            self.prev_data = self.get_prev_row()
            # 만약, 초기화 상태라면 prev_time = self.simulation_start_time
            # 이미 simulation 을 진행하고 있었더라면, prev_time 은 simulated_time 으로 결정

            # prev_data 로 확인된, simulation 상에서 진행된 시각.
            simulated_time = str(self.prev_data.time.item())

            # 이미 형성된 DB 에서, 진행된 시각이 현재시각보다 같을때 까지, 반복문을 skip
            if str(simulated_time) >= str(self.current_time):
                self.skip = True
                return None

            # 이전 행을 기준으로 현재 시간에 대한 db 생성.
            # 결국, self.current_time 에 대한 데이터를 이번에 쌓은 과정.
        elif not self.save_:
            self.prev_data = self.total_df_for_db.tail(1).reset_index(drop=True).copy()

        self.current_total_db = self.prev_data.copy()
        self.current_total_db['time'] = self.current_time

        self.skip = False

    def decision_buy(self):
        if not self.future:
            self.current_transaction_amount = round(self.buy_ratio * self.init_balance, 0)
            if self.current_available < self.current_transaction_amount:
                self.decision = 'stay'
            else:
                self.current_amount = (self.current_transaction_amount / self.current_price) * (1 - self.fee)
                self.update_amount = self.prev_amount + self.current_amount
                self.update_available = self.current_available - self.current_transaction_amount
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
                    realized_gain = np.abs(realized_gain) * (1-self.fee) * np.sign(realized_gain)
                    # 지금 양 만큼 이전 계약의 회수
                    realized_contract = coin_contract_amount * self.current_price
                    realized_contract = np.abs(realized_contract) * (1-self.fee) * np.sign(realized_contract)
                    # 닫아진 계약만큼의 손익과 계약금 회수
                    self.update_available = self.current_available + realized_gain + realized_contract

                # 공매도 계약에 대한 포지션을 모두 닫고, 일부 남은 공매수 포지션을 열기.
                else:
                    # 공매도 포지션을 닫고, 이전 계약금 소거 및 새로운 계약 만큼 공매수 포지션 생성
                    self.update_contract_amount = np.abs(self.prev_contract_amount - coin_contract_amount)
                    self.update_amount = self.update_contract_amount * self.leverage
                    self.update_executed = self.current_price
                    # 공매도 포지션 종료에 따른 손익. self.prev_amount ( < 0) 만큼 포지션이 종료
                    realized_gain = self.prev_amount * (self.current_price - self.prev_executed)
                    realized_gain = np.abs(realized_gain) * (1-self.fee) * np.sign(realized_gain)
                    # 이전 포지션 계약의 전량 회수
                    realized_contract = self.prev_contract_amount * self.current_price - \
                                        self.update_contract_amount * self.current_price
                    realized_contract = np.abs(realized_contract) * (1-self.fee) * np.sign(realized_contract)
                    # 닫아진 계약만큼의 손익과 계약금 회수
                    self.update_available = self.current_available + realized_gain + realized_contract

                # 포지션 전체 종료 경우
                if self.update_amount == 0:
                    self.update_executed = 0
                    self.rate = 0
                    self.position = 'close'
                # update aomunt 에 따라 최종 포지션 결정.
                elif self.update_amount < 0:
                    self.position = 'short'
                    self.rate = - (self.current_price - self.update_executed) / self.update_executed * 100
                elif self.update_amount > 0:
                    self.position = 'long'
                    self.rate = (self.current_price - self.update_executed) / self.update_executed * 100

            # 공매수 포지션 추가
            else:
                self.update_amount = self.prev_amount + self.current_amount
                self.update_available = self.current_available - contract_amount
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
            self.limit_order_add(self.trading_ticker, self.current_amount,
                                 self.current_time, self.limit_low, self.limit_high)

    def decision_sell(self):
        # 전량매도.
        if not self.future:
            if self.prev_amount == 0:
                self.decision = 'stay'
            elif self.sell_ratio != 1:
                self.current_transaction_amount = self.prev_amount * self.sell_ratio * self.prev_executed
                sold_amount = self.current_transaction_amount * (1 - self.fee)
                self.update_available = self.current_available + sold_amount
                self.update_executed = self.prev_executed
                self.update_amount = self.prev_amount * (1 - self.sell_ratio)
                self.position = "long"
                self.rate = (self.current_price - self.update_executed) / self.update_executed * 100
            else:
                self.current_transaction_amount = self.prev_amount * self.prev_executed
                sold_amount = self.current_transaction_amount * (1 - self.fee)
                self.update_available = self.current_available + sold_amount
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
                    realized_gain = np.abs(realized_gain) * (1-self.fee) * np.sign(realized_gain)
                    # 지금 양 만큼 이전 계약의 회수
                    realized_contract = coin_contract_amount * self.current_price
                    realized_contract = np.abs(realized_contract) * (1-self.fee) * np.sign(realized_contract)
                    # 닫아진 계약만큼의 손익과 계약금 회수
                    self.update_available = self.current_available + realized_gain + realized_contract

                # 공매수 계약에 대한 포지션을 모두 닫고, 일부 남은 공매도 포지션을 열기.
                else:
                    # 공매수 포지션을 닫고, 이전 계약금 소거 및 새로운 계약 만큼 공매도 포지션 생성.
                    self.update_contract_amount = np.abs(self.prev_contract_amount - coin_contract_amount)
                    self.update_amount = - self.update_contract_amount * self.leverage
                    self.update_executed = self.current_price
                    # 공매수 포지션 종료에 따른 손익. self.prev_amount( > 0 ) 만큼 포지션이 종료
                    realized_gain = self.prev_amount * (self.current_price - self.prev_executed)
                    realized_gain = np.abs(realized_gain) * (1- self.fee) * np.sign(realized_gain)
                    # 이전 포지션 계약의 전량 회수
                    realized_contract = self.prev_contract_amount * self.current_price - \
                                        self.update_contract_amount * self.current_price
                    realized_contract = np.abs(realized_contract) * (1-self.fee) * np.sign(realized_contract)
                    # 닫아진 계약만큼의 손익과 계약금 회수
                    self.update_available = self.current_available + realized_gain + realized_contract

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
                self.update_available = self.current_available - contract_amount
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
            self.limit_order_add(self.trading_ticker, self.current_amount,
                                 self.current_time, self.limit_low, self.limit_high)

    def decision_stay(self):
        self.update_available = self.current_available
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
            # self.prev_amout < 0 then self.current_price 가 낮아야 이득.
            # self.prev_amout > 0 then self.current_price 가 높아야 이득.
            realized_gain = self.prev_amount * (self.current_price - self.prev_executed)
            realized_gain = np.abs(realized_gain) * (1-self.fee) * np.sign(realized_gain)
            # 이전 계약 전체회수
            realized_contract = self.prev_contract_amount * self.current_price
            realized_contract = np.abs(realized_contract) * (1-self.fee) * np.sign(realized_contract)
            self.update_available = self.current_available + realized_gain + realized_contract
            self.position = 'close'
        else:
            self.decision = 'stay'

    def liquidation_check(self):
        if self.future:
            # self.current_price > self.update_executed : self.update_amount > 0 이면 이득
            # self.current_price < self.update_executed : self.update_amount < 0 이면 이득

            if self.rate < 0 and self.update_contract_amount < np.abs(self.update_amount * (self.rate / 100)):
                print(f"{self.trading_ticker} 청산.")
                self.update_contract_amount = 0
                self.position = 'close'
                self.rate = 0

    def create_ticker_update_db(self, ticker):
        self.update_db = defaultdict(list)
        self.update_db["time"] = [self.current_time]
        self.update_db["available"] = [float(self.update_available)]
        self.update_db["leverage"] = [float(self.leverage)]
        self.update_db[f"current_{ticker}_amount"] = [self.update_amount]
        self.update_db[f"current_{ticker}_price"] = [self.current_price]
        self.update_db[f"executed_{ticker}_price"] = [self.update_executed]
        self.update_db[f"{ticker}_decision"] = [self.decision]
        self.update_db[f"{ticker}_position"] = [self.position]
        self.update_db[f"{ticker}_rate"] = [self.rate]
        self.update_db[f"{ticker}_contract_amount"] = [self.update_contract_amount]

        # self.update_db["net_worth"]=[update_net_worth]
        self.update_db = pd.DataFrame.from_dict(dict(self.update_db))

        if not self.future:
            # 총 평가 금액
            self.total_eval += np.abs(self.update_amount * self.current_price)
        else:
            self.total_contract += self.update_contract_amount * self.current_price

        self.current_total_db.update(self.update_db)

    # net_worth
    def update_net_worth(self):
        update_net_worth = self.current_total_db['available'].item() + self.total_eval \
            if not self.future else self.current_total_db["available"].item() + self.total_contract
        update_net_worth_db = pd.DataFrame([update_net_worth], columns=['net_worth'])
        self.current_total_db.update(update_net_worth_db)

    def saving_row(self):
        if self.save_:
            print(f"{self.current_time} Decision - To DB ")
            self.current_total_db.to_sql(self.simulation_table_name,
                                         sql_connect('simulator'), index=False, if_exists='append')
        else:
            print(f"{self.current_time} Decision - To DataFrame ")
            self.total_df_for_db = self.total_df_for_db.append(self.current_total_db, ignore_index=True)

    # 실제 trading simulation 부분
        # 현재가 - 종가.
        # 수수료 계산 - amount 를 적게 처리. - 매수
        # 거래 후 정산금액을 적게 처리 - 매도.
    def simulation_trading(self):

        print("Setting Simulator DB ...")
        self.simulator_table_setting()
        if self.limit:
            self.make_limit_order_db()
        print("Done")

        print(f"Tickers : {self.ticker_list_for_db}")

        # 한번에 저장하는 방식을 결정한 경우, df 에 대한 초기화가 최초에 한하여 필요.
        if not self.save_:
            self.total_df_for_db = self.init_simulation_df

            if self.limit:
                self.total_limit_order_df_for_db = self.init_order_list_df

        n = self.observed_rows
        # self.current_time 을 가지는 simulation row 를 DB 에 쌓아가며 simulation 을 진행.
        for i in range(len(self.total_df) - n):
            # initializing 이 index 0 을 기준으로 start time 을 지정에 유의
            idx = i + 1
            data = self.total_df[idx:idx + n]

            # 시간 설정 및 이전 열 데이터 불러오기, rows 마다 필요한 초기화 설정
            self.current_time = data.tail(1).time.item()
            self.setting_prev_data()
            if self.skip:
                continue
            self.init_trader_variables_setting()

            print(f"{self.current_time} Decision - ticker_list ")

            for ticker in self.ticker_list_for_db:

                # TimeChecker start
                start_time = time.time()

                self.trading_ticker = ticker
                self.init_trader_variables_setting_ticker()
                self.init_limit_setting_ticker()
                self.setting_trader_ticker_df(ticker, data)
                if self.early_stopping(): break

                # TimeChecker fin
                print(f"Prev Data checker time {time.time() - start_time}")

                # 거래를 시작하기 전, limit order 에 대해 계산할것을 고려
                # 업데이트가 필요한 변수들 : prev_amount, prev_contract_amount, current_available

                # TimeChecker start
                start_time = time.time()

                if self.limit:
                    if not self.save_:
                        self.total_limit_order_df_for_db.reset_index(drop=True)
                    else:
                        self.get_limit_order_db()

                    self.limit_order_db_check(ticker, self.prev_executed)
                    if self.limit_total_gain != 0.0:
                        print(f"Limit Order at {self.total_order_executed_time} be Signed")
                        print(ticker, self.prev_amount, self.limit_total_amount)
                        self.prev_amount -= self.limit_total_amount
                        if self.prev_amount == 0:
                            self.prev_executed = 0
                        if self.future:
                            update_contract_amount = np.abs(self.prev_amount / self.leverage)
                            current_realized_contract = (self.prev_contract_amount - update_contract_amount) * self.current_price
                            current_realized_contract = np.abs(current_realized_contract) * (1-self.fee)
                            self.prev_contract_amount = update_contract_amount
                            self.limit_total_gain = np.abs(self.limit_total_gain) * (1-self.fee) * np.sign(self.limit_total_gain)
                            total_realized_gain = self.limit_total_gain + current_realized_contract
                            print(self.limit_total_gain, current_realized_contract)
                            print(total_realized_gain)
                            self.current_available += total_realized_gain
                        else:
                            self.current_available += np.abs(self.limit_total_transaction) * (1 - self.fee) * np.sign(self.limit_total_transaction)

                # TimeChecker fin
                print(f"limit checker time {time.time() - start_time}")

                # TimeChecker start
                start_time = time.time()

                if self.decision == 'buy':
                    self.decision_buy()

                elif self.decision == 'sell':
                    self.decision_sell()

                if self.decision == 'close':
                    self.decision_close()

                if self.decision == 'stay':
                    self.decision_stay()

                self.liquidation_check()
                self.create_ticker_update_db(ticker)

                # TimeChecker fin
                print(f"Decision checker time {time.time() - start_time}")

            # TimeChecker start
            start_time = time.time()

            self.update_net_worth()
            self.saving_row()
            if self.early_stopping():
                break

            # TimeChecker fin
            print(f"Update checker time {time.time() - start_time}")


        if not self.save_:
            self.total_df_for_db.to_sql(self.simulation_table_name,
                                    sql_connect('simulator'), index=False, if_exists='replace')

# TODO
# 시간이 너무 많이드는 부분 - 최소한의 connection 을 limit process 에 적용할 수 있는가 ?
if __name__ == "__main__":
    start_time = time.time()
    s = Simulator(1000000, 0)
    s.future_setting()
    s.init_simulation_whether()
    s.save_setting()
    s.limit_order_setting()
    s.leverage = 10

    # s.algorithm_connection(s.get_ticker_data_from_db(20210407, 20210408, ["random", 1]))
    s.get_ticker_data_from_db(20210407, 20210408, ["btckrw", "ethkrw"])
    s.algorithm_connection(algo=LimitAlgorithm)
    s.simulation_trading()

    print(f"time {time.time()-start_time}")
    # 1440분 10 종목 시뮬 시간 ~ 190초 ~~ 7일 25분정도,,
