from mysql_func import *

import pandas as pd
import pyupbit
from collections import defaultdict
import requests
from bs4 import BeautifulSoup

import json
from datetime import datetime, timedelta


class Collector:
    def __init__(self):

        self.db_conn = db_connection()
        self.setting_engine = None
        self.db_engine = None

        self.db_exist_check()
        self.update_check_ticker()

    # 기본적으로 DB 들이 존재하는지 확인 및 각 DB 별 연결 생성.
    def db_exist_check(self):
        db_check("coin_min_db")
        db_check("setting")

        self.db_engine = sql_connect("coin_min_db")
        self.setting_engine = sql_connect("setting")

        self.ticker_list_init()
        self.db_update_checker_init()

    @staticmethod
    # Ticker list 초기화 함수. replace.
    def ticker_list_init():
        ticker_data = pyupbit.get_tickers()
        tickers_df = pd.DataFrame(ticker_data, columns=['ticker'])
        tickers_df.to_sql("ticker_list", sql_connect("setting"), index=False, if_exists='replace')

    @staticmethod
    # Update 여부를 확인할 수 있는 DB TABLE 생성. 있을경우 pass
    def db_update_checker_init():
        if not table_exist("setting", "update_checker"):
            tickers = defaultdict(list)
            ticker_data = pyupbit.get_tickers()
            iteration = 0
            # make dict : {1 : [ticker1, 0, 0],
            #              2 : [ticker2, 0,0],
            #              ...
            #              }
            for i in ticker_data:
                tickers[iteration] = [i, 0, 0]
                iteration += 1
            tickers_df = pd.DataFrame.from_dict(dict(tickers), orient='index',
                                                columns=['ticker', 'update_date', 'update_date_unix'])
            tickers_df.to_sql("update_checker", sql_connect("setting"), index=False)
        else:
            pass

    # 생성된 update_checker 에서 특정 ticker 를 받아와서, 최종 업데이트 시각을 가져옴.
    # updated 된 시각을 저장. 기준을 KRW-BTC 로 잡음
    def get_last_update_date(self):
        sql = "SELECT update_date_unix FROM setting.update_checker WHERE ticker='KRW-BTC'"
        last_date = self.setting_engine.execute(sql).fetchall()[0][0]

        return last_date

    # Ticker list 와 Update checker 의 Tickers 를 비교해서, 추가된 ticker append.
    # 추가된 ticker 를 update_checker 의 기준 ticker 의 update_date 를 참고하여 update_checker 를 업데이트.
    def update_check_ticker(self):
        sql = "SELECT list.ticker FROM setting.ticker_list list WHERE list.ticker NOT IN " \
              "(SELECT checker.ticker FROM setting.update_checker checker)"
        ticker_update = self.setting_engine.execute(sql).fetchall()
        update_tickers = defaultdict(list)
        iteration = 0
        if ticker_update is not None:
            last_date = self.get_last_update_date()
            for i in ticker_update:
                update_tickers[iteration] = [i[0], 0, last_date]
                iteration += 1
            tickers_df = pd.DataFrame.from_dict(dict(update_tickers), orient='index',
                                                columns=['ticker', 'update_date', 'update_date_unix'])
            tickers_df.to_sql("update_checker", sql_connect("setting"), index=False, if_exists="append")
        else:
            pass

    # Collecting 할 때 어느시점 까지 이전에 이루어 졌는지, 즉 업데이트가 얼마나 되었는지 확인.
    # 업데이트가 필요한지 아닌지 반환
    def update_need_checker(self, ticker, start_time, fin_time):
        sql = f"select update_date_unix from setting.update_checker where ticker='{ticker}'"
        date_unix = self.setting_engine.execute(sql).fetchall()
        if date_unix[0][0] == fin_time:
            # Update_checker 의 unix 타임과 현재 시간을 비교.
            # fin_time 이 date_unix_time 와 동일하면 콜렉팅 완료.
            return False, fin_time
        elif date_unix[0][0] == 0:
            return True, start_time
        else:
            return True, date_unix[0][0]

    def update_start_time(self, init_symbol, start_time, fin_time):
        # 받아온 시간을 unix_time 으로 변경
        start_time_unix = int(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp())
        fin_time_unix = int(datetime.strptime(fin_time, "%Y-%m-%d %H:%M:%S").timestamp())

        # 각 ticker 에 대해, update_checker DB 를 통해 크롤링 필요여부를 확인.
        # update 를 시작할 날짜를 update_checker 의 기준 ticker 날짜로 설정.
        update_check, date_unix = self.update_need_checker(init_symbol, start_time_unix, fin_time_unix)

        # 업데이트가 필요한 경우, start_time_unix 를 업데이트가 진행된 곳 바로 다음 분 단위로 설정.
        # print 는 한국시간 단위이기 때문에, +9h 를 하여 출력.
        if update_check:
            db_date = datetime.strftime(datetime.utcfromtimestamp(date_unix + 9 * 60 * 60), "%Y%m%d")
            print(f"{init_symbol} MIN DB 가 {db_date}까지 이미 존재. Continue")
            start_time_unix = date_unix + 60
            return start_time_unix, fin_time_unix

        else:
            db_date = datetime.strftime(datetime.utcfromtimestamp(date_unix + 9 * 60 * 60), "%Y%m%d")
            print(f"{init_symbol} MIN DB 가 {db_date}까지 업데이트 되어있음. Continue")
            return None, None

    @staticmethod
    def get_data_from_url(symbol, start_time_unix, iteration, divide_int):
        start_time = start_time_unix + iteration * divide_int
        fin_time = start_time_unix + (iteration + 1) * divide_int
        start_time = int(start_time)
        fin_time = int(fin_time)
        url = f"https://crix-api-tv.upbit.com/v1/crix/tradingview/history?" \
              f"symbol={symbol}&resolution=1&from={int(start_time)}&to={int(fin_time)}"

        # unix_time 을 간격으로 하여 url 접속 및 데이터를 df 형태로 받아오기.

        res = requests.get(url)
        html = res.text

        soup = BeautifulSoup(html, 'lxml')
        result = soup.find('body').text
        hist_dict = json.loads(result)

        return hist_dict, url, fin_time

    # noinspection DuplicatedCode
    @staticmethod
    def columns_setting(hist_dict):
        hist_df = pd.DataFrame.from_dict(hist_dict)
        hist_df['t'] = pd.to_datetime(hist_df['t'], unit='ms')  # raw data 가 ms 기준으로 되어있음에 유의.
        hist_df['t'] = pd.DatetimeIndex(hist_df['t']) + timedelta(hours=9)  # 한국시간으로변경
        hist_df = hist_df.rename(
            columns={"t": "time", "o": "open", "l": "low", "h": "high", "c": "close", "v": "volume"})

        return hist_df

    def data_processing(self, hist_dict):
        # DATAFRAME 생성

        hist_df = self.columns_setting(hist_dict)

        hist_df['close'] = hist_df.close.fillna(method='ffill')
        hist_df['open'] = hist_df.open.fillna(hist_df['close'])
        hist_df['low'] = hist_df.low.fillna(hist_df['close'])
        hist_df['high'] = hist_df.high.fillna(hist_df['close'])
        # fill to prior close
        hist_df['volume'] = hist_df.volume.fillna(0)
        hist_df = hist_df.fillna(0)
        return hist_df

    def update_db(self, symbol, init_symbol, hist_df, fin_time):
        # DB에 저장 및, update_checker 의 최종 업데이트 시각 업데이트.
        # conn 이 많이 쓰이지만, crawling 의 저장 안정성을 위해 불가피.

        hist_df.to_sql(str(symbol).lower(), self.db_engine, if_exists='append', index=False)
        print(f"{symbol} 테이블 생성완료!")

        fin_time_datetime = datetime.utcfromtimestamp(fin_time).strftime('%Y%m%d')
        sql = f"UPDATE setting.update_checker set update_date_unix = {fin_time} WHERE ticker = '{init_symbol}'"
        self.db_conn.cursor().execute(sql)
        self.db_conn.commit()
        sql = f"UPDATE setting.update_checker set update_date = {fin_time_datetime} WHERE ticker = '{init_symbol}'"
        self.db_conn.cursor().execute(sql)
        self.db_conn.commit()
        print(f"{symbol} set update_date : {fin_time_datetime}")

    # 특정 Symbol(ticker) 에 대한 분별 데이터 크롤링 및 DB 저장
    def coin_min_craw(self, symbol, init_symbol, start_time, fin_time):
        symbol = symbol.replace(" ", "")
        start_time_unix, fin_time_unix = self.update_start_time(init_symbol, start_time, fin_time)
        if start_time_unix and fin_time_unix is not None:
            # 업데이트가 덜된 부분을 하루단위로 craw

            # unix time difference to day_ delta
            time_delta_day = int((fin_time_unix - start_time_unix) / (60 * 60 * 24))
            if time_delta_day == 0:
                time_delta_day = 1
            divide_int = (fin_time_unix - start_time_unix) / time_delta_day

            # 업데이트까지 완전히 되기위해 time_delta_day 만큼 일 단위의 업데이트가 필요.
            # 반복문을 통해 반복 craw & DB save
            for i in range(int(time_delta_day)):

                hist_dict, url, fin_time = self.get_data_from_url(symbol, start_time_unix, iteration=i,
                                                                  divide_int=divide_int)

                if hist_dict['s'] == 'ok':
                    del (hist_dict['s'])
                    print(f"{symbol} 테이블 생성중")

                elif hist_dict['s'] == "no_data":
                    print("data 없음. Continue", end='\r')
                    continue

                else:
                    print(f"Error Occur, not Satisfied URL condition, \n With {start_time}&{fin_time}")
                    print(url)

                hist_df = self.data_processing(hist_dict)
                self.update_db(symbol, init_symbol, hist_df, fin_time)

        # replace : 삭제 후 재생성. #append : 추가. #fail : 있을경우, continue

    # ticker 를 craw 를 위한 형태로 변환
    @staticmethod
    def ticker_trans_for_craw(ticker):
        init_ticker = ticker
        string = ticker.replace("-", " ")
        split_strings = string.split()
        reversed_split_strings = list(reversed(split_strings))
        ticker = ' '.join(reversed_split_strings)

        return init_ticker, ticker

    # coin_min_craw 를 ticker 에 대해 반복, 시작 및 완료 시점 설정.
    def min_craw_db(self, start_time, fin_time):
        # 1일치마다 craw

        ticker_list = pyupbit.get_tickers()

        for ticker in ticker_list:
            init_ticker, ticker = self.ticker_trans_for_craw(ticker)
            self.coin_min_craw(ticker, init_ticker, start_time, fin_time)

    # 데이터 입력이 오류로 인해 잘못되었을 경우, 완전히 Ticker DB 를 새로 받기 위해서는,
    # update_checker 에서 해당 ticker 의 update_date_time 을 0 으로 변경하고, table 삭제 후 .py 실행

    def min_craw_db_ticker(self, start_time, fin_time, ticker):
        init_ticker, ticker = self.ticker_trans_for_craw(ticker)
        self.coin_min_craw(ticker, init_ticker, start_time, fin_time)



if __name__ == '__main__':

    collector = Collector()
    # 인풋은 반드시 %Y-%m-%d 00:00:00 타입의 시작과 끝값.

    now = datetime.now().strftime('%Y-%m-%d 00:00:00')

    start_ = "20170101"
    # start type transform to %Y-%m-%d 00:00:00 form
    start_ = datetime.strptime(start_, '%Y%m%d')
    start_ = start_.strftime('%Y-%m-%d 00:00:00')
    collector.min_craw_db(start_, now)
