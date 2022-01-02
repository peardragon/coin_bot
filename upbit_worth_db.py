import pyupbit
import private_key as key
import datetime
import pandas as pd
from mysql_func import db_check, create_engine, sql_connect
import time

upbit = pyupbit.Upbit(key.upbit_access, key.upbit_secret)
db_check("upbit_balance")
db_engine = sql_connect("upbit_balance")

while True:
    net_worth = upbit.get_balances()
    print(net_worth)
    total_balance = 0
    KRW = 0
    ticker = 0
    for items in net_worth:
        if items["currency"] == "KRW":
            KRW = (float(items['balance'])+float(items["locked"]))
            total_balance += KRW
        else:
            current_price = pyupbit.get_current_price(f"KRW-{items['currency']}")
            ticker += (float(items['balance'])+float(items["locked"])) * float(current_price)
            total_balance += ticker

    current_balance = pd.DataFrame([[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                     KRW, ticker, total_balance]])
    current_balance.columns = ["Time", "KRW", "Ticker", "Net worth"]
    print(current_balance)
    current_balance.to_sql("net_worth", db_engine, if_exists='append', index=False)
    time.sleep(60)

