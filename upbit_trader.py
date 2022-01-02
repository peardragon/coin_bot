import pyupbit
# from min_algorithm import restore_model
from min_craw_2 import Collector
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import datetime
import private_key as key
import pickle
import glob
from lightgbm_model import CoinObjective

def restore_model(filename):
    for file in glob.glob(filename):
        with open(file, 'rb') as model:
            print("restore model in : ", file)
            rate = float(file[file.find("rate_") + 5:file.find(".sav")])
            loaded_model = pickle.load(model)
            break
    return loaded_model, rate

def restore_optimized_model(filename):
    for file in glob.glob(filename):
        with open(file, 'rb') as model:
            print("restore model in : ", file)
            rate = float(file[file.find("rate_") + 5:file.find(".sav")])
            interval = float(file[file.find("interval_") + 9:file.find("_rate")])
            loaded_model = pickle.load(model)
            break
    return loaded_model, interval, rate

def order_list_sync():
    complement = [i for i in [j for j in limit_order_uuid]
                  if i not in [k["uuid"] for k in upbit.get_order(ticker)]]

    if len(complement) != 0:
        for item in complement:
            idx = limit_order_uuid.index(item)

            uuid = limit_order_uuid[idx]
            volume = limit_order_volume[idx]
            lower_price = limit_order_lower_price[idx]
            upper_price = limit_order_upper_price[idx]
            order_time = limit_order_time[idx]

            limit_order_uuid.remove(uuid)
            limit_order_volume.remove(volume)
            limit_order_upper_price.remove(upper_price)
            limit_order_lower_price.remove(lower_price)
            limit_order_time.remove(order_time)




if __name__ == "__main__":
    # wm = WebSocketManager("ticker", ["KRW-BTC"])
    # data = wm.get()
    # print(data)
    # # wm.terminate()
    upbit = pyupbit.Upbit(key.upbit_access, key.upbit_secret)
    # custom
    # config = dict(time="20201010_20201020", interval=80, rate=0.7)
    while True:

        # ## Default Algorithm Config ## #
        fin_time_auto = datetime.datetime.now().date()
        start_time_auto = fin_time_auto - datetime.timedelta(days=6)

        fin_time_auto = fin_time_auto.strftime("%Y%m%d")
        start_time_auto = start_time_auto.strftime("%Y%m%d")
        # start time date 00:00:00 ~ fin time date 00:00:00

        # time_config = start_time_auto + "_" + fin_time_auto
        # interval_config = 60
        # rate_config = 0.55
        ticker_config = "btckrw"

        ticker = "KRW-BTC"
        limit_upper = limit_lower = None

        trader_start_time = datetime.datetime.now().date().strftime("%Y%m%d")
        collector = Collector()
        now = datetime.datetime.now().strftime('%Y-%m-%d 00:00:00')
        start_db = "20170101"
        # start type transform to %Y-%m-%d 00:00:00 form
        start_db = datetime.datetime.strptime(start_db, '%Y%m%d')
        start_db = start_db.strftime('%Y-%m-%d 00:00:00')
        collector.min_craw_db_ticker(start_db, now, ticker)

        # config = dict(time=time_config, interval=int(interval_config))

        # Model Connection
        try:
            model, interval, rate = restore_optimized_model(f'./lgbm_model/*_{fin_time_auto}/'
                                                            f'optimized_model_interval_*_rate_*.sav')
        except:
            print("Model un-Exist. Make model")
            import pandas as pd
            from lightgbm_model import train_data_processing, get_sql_data, Objective, \
                restore_model_best_params, train_test_split, createFolder
            import optuna

            study = optuna.create_study(direction="maximize")
            fin_time = datetime.datetime.now().date()
            study.optimize(CoinObjective(fin_time), n_trials=500, timeout=600)

            best_clf_ = restore_model_best_params(study.best_trial.number)
            days = study.best_params['days']
            interval = study.best_params['interval']
            rate = round(study.best_params['rate'], 3)

            start = (fin_time - datetime.timedelta(days=days)).strftime("%Y%m%d")
            finish = fin_time.strftime("%Y%m%d")
            res = get_sql_data("btckrw", start, finish)

            date_path = start + "_" + finish + "/"
            directory_path = "./lgbm_model/" + date_path
            configs = "_interval_" + str(interval) + "_rate_" + str(rate)
            model_name = "optimized_model" + configs + ".sav"
            file_path = "./lgbm_model/" + date_path + model_name

            ticker_data = pd.DataFrame(res, columns=['time', 'open', 'low', 'high', 'close', 'volume'])
            ticker_data["interval_max"] = ticker_data.high[::-1].rolling(interval).max()[::-1]
            ticker_data = ticker_data[interval:].reset_index(drop=True)

            X, y = train_data_processing(res, interval, rate)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

            best_clf_.fit(X_train, y_train, verbose=2)

            print("Optimize and Fit Finish : Save model")

            createFolder(directory_path)

            with open(file_path, 'wb') as file:
                pickle.dump(best_clf_, file)

            model, interval, rate = restore_optimized_model(f'./lgbm_model/*_{fin_time_auto}/'
                                                            f'optimized_model_interval_*_rate_*.sav')

        print("Algorithm Connect")

        # Log : Order List Connection
        try:
            with open('./upbit log/log.txt', 'rb') as file:
                total_log = pickle.load(file)

            limit_order_uuid = total_log[0]

            limit_order_lower_price = total_log[1]
            limit_order_upper_price = total_log[2]

            limit_order_volume = total_log[3]
            # initial order time
            limit_order_time = total_log[4]
        except FileNotFoundError:
            limit_order_uuid = []

            limit_order_lower_price = []
            limit_order_upper_price = []

            limit_order_volume = []
            limit_order_time = []

        # Trading Part
        while True:
            current_time = datetime.datetime.now().date().strftime("%Y%m%d")
            X = []

            # Decision Maker
            try:
                df = pyupbit.get_ohlcv("KRW-BTC", count=int(interval) + 1, interval="minute1")
                # Data Arr Configuration
                # data_arr = df[["open", "high", "volume"]].iloc[:-1].to_numpy()
                data_arr = df[["open", "low", "close", "high", "volume"]].iloc[:-1].to_numpy()
                scaled_data = MinMaxScaler().fit_transform(X=data_arr)
                X.append(scaled_data.flatten())
                X = np.asarray(X, dtype=float)
            except (TypeError, IndexError):
                continue

            pred = model.predict(X)
            print("Current Time: ", datetime.datetime.now())
            print("Decision: ", pred[0])
            current_open_price = df["open"].iloc[-1]
            prev_open_price = df["open"].iloc[-2]

            # True Decision Part
            if pred[0] == 1:
                # 매수 금액은 수수료를 제외한 금액입니다
                # order = upbit.buy_market_order(ticker, 10000)
                order = upbit.buy_limit_order(ticker, current_open_price, round(10000 / current_open_price, 8))
                decision_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print("Current Executed order : ", order)
                print("Current Open Price: ", current_open_price)
                current_price = pyupbit.get_current_price(ticker)
                print("Current Price: ", current_price)

                if "error" in order:
                    pass
                else:
                    while True:
                        try:
                            print("Upbit Waiting for buy Order : ", upbit.get_order(order["uuid"]), end="\r")
                            if (datetime.datetime.now() - datetime.timedelta(minutes=1)) > \
                                    datetime.datetime.strptime(decision_time, "%Y-%m-%d %H:%M:%S"):
                                print("Order failed. Skip this Order")
                                upbit.cancel_order(order["uuid"])
                                break

                            if upbit.get_order(order["uuid"])["state"] == "done":
                                print("Order Conducted")
                                print("Upbit Conducted buy Order : ", upbit.get_order(order["uuid"]))
                                volume = upbit.get_order(order["uuid"])["executed_volume"]
                                limit_upper_price = round(current_price * (1 + rate / 100), -3)
                                limit_lower_price = round(current_price * (1 - rate / 100), -3)
                                limit_upper = upbit.sell_limit_order(ticker, limit_upper_price, volume)
                                print("Current Executed limit order : ", limit_upper)
                                if "error" in limit_upper:
                                    print("Current Volume : ", upbit.get_balances())
                                    volume = upbit.get_balances()[1]["balance"]
                                    limit_upper = upbit.sell_limit_order(ticker, limit_upper_price, volume)
                                    print("주문 재설정 : ", ticker, volume)
                                    continue

                                limit_order_uuid.append(limit_upper["uuid"])

                                limit_order_lower_price.append(limit_lower_price)
                                limit_order_upper_price.append(limit_upper_price)
                                limit_order_volume.append(volume)

                                limit_order_time.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                break
                            print("Wait for Order Signed ...", end="\r")
                            time.sleep(0.5)

                        except (TypeError, IndexError, KeyError):
                            time.sleep(0.5)

            # 손절 및 익절 주문 변환

            prev_open_price = current_open_price
            for i in range(50):
                current_open_price = pyupbit.get_current_price(ticker)
                min_rate = (current_open_price - prev_open_price) / prev_open_price * 100
                print("Current Time: ", datetime.datetime.now())
                print("Min rate : ", min_rate, "%")
                orders = pyupbit.get_orderbook(ticker)

                for limit_lower_price in limit_order_lower_price:
                    order_prices = sum([[order["ask_price"], order["bid_price"]]
                                        for order in orders[0]["orderbook_units"]], [])
                    order_prices.sort()
                    if limit_lower_price in order_prices or limit_lower_price > order_prices[0]:
                        idx = limit_order_lower_price.index(limit_lower_price)

                        uuid = limit_order_uuid[idx]
                        volume = limit_order_volume[idx]
                        order_time = limit_order_time[idx]
                        limit_upper_price = limit_order_upper_price[idx]
                        limit_lower_price = limit_order_lower_price[idx]

                        # if (datetime.datetime.now() - datetime.timedelta(minutes=interval)) \
                        #         < datetime.datetime.strptime(order_time, "%Y-%m-%d %H:%M:%S") \
                        #         and min_rate > -rate/2:
                        #     continue

                        # Before order limit upper order check - skip
                        if upbit.get_order(uuid)["price"] == str(limit_lower_price):
                            continue


                        limit_order_uuid.remove(uuid)
                        limit_order_volume.remove(volume)
                        limit_order_time.remove(order_time)
                        limit_order_upper_price.remove(limit_upper_price)
                        limit_order_lower_price.remove(limit_lower_price)

                        # Sync
                        if upbit.get_order(uuid)["state"] == "done":
                            continue

                        # 이전에 limit upper order 을 취소하고, limit lower order 을 다시 설정.
                        upbit.cancel_order(uuid)
                        while True:
                            if upbit.get_order(uuid)["state"] == "cancel":
                                break
                            print("Wait for Order Canceled ...", end="\r")
                            time.sleep(0.5)

                        limit_lower = upbit.sell_limit_order(ticker, limit_lower_price, volume)
                        print("손절 주문 설정 : ", ticker, volume)
                        print("Canceled Order: ", upbit.get_order(uuid))
                        print("Make Lower Limit Order : ", limit_lower)

                        if "error" in limit_lower:
                            print("Current Volume : ", upbit.get_balances())
                            volume = upbit.get_balances()[1]["balance"]
                            limit_lower = upbit.sell_limit_order(ticker, limit_lower_price, volume)
                            print("손절 주문 설정 : ", ticker, volume)
                            print("Canceled Order: ", upbit.get_order(uuid))
                            print("Make Lower Limit Order : ", limit_lower)

                        # 현 주문에 대한 데이터 log 에 재저장.

                        limit_order_uuid.append(limit_lower["uuid"])

                        limit_order_lower_price.append(limit_lower_price)
                        limit_order_upper_price.append(limit_upper_price)
                        limit_order_volume.append(volume)

                        # limit_order_time.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        limit_order_time.append(order_time)

                orders = pyupbit.get_orderbook(ticker)

                for limit_upper_price in limit_order_upper_price:
                    order_prices = sum([[order["ask_price"], order["bid_price"]]
                                        for order in orders[0]["orderbook_units"]], [])
                    order_prices.sort()
                    if limit_upper_price in order_prices or order_prices[-1] > limit_upper_price:
                        idx = limit_order_upper_price.index(limit_upper_price)

                        uuid = limit_order_uuid[idx]
                        volume = limit_order_volume[idx]
                        order_time = limit_order_time[idx]
                        limit_lower_price = limit_order_lower_price[idx]
                        limit_upper_price = limit_order_upper_price[idx]

                        # if (datetime.datetime.now() - datetime.timedelta(minutes=interval)) \
                        #         < datetime.datetime.strptime(order_time, "%Y-%m-%d %H:%M:%S")\
                        #         and min_rate > -rate/2:
                        #     continue

                        # Before order limit upper order check - skip
                        if upbit.get_order(uuid)["price"] == str(limit_upper_price):
                            continue

                        limit_order_uuid.remove(uuid)
                        limit_order_volume.remove(volume)
                        limit_order_time.remove(order_time)
                        limit_order_lower_price.remove(limit_lower_price)
                        limit_order_upper_price.remove(limit_upper_price)

                        # Sync
                        if upbit.get_order(uuid)["state"] == "done":
                            continue

                        # 이전에 limit lower order 을 취소하고, limit upper order 을 다시 설정.
                        upbit.cancel_order(uuid)
                        while True:
                            if upbit.get_order(uuid)["state"] == "cancel":
                                break
                            print("Wait for Order Canceled ...", end="\r")
                            time.sleep(0.5)

                        limit_upper = upbit.sell_limit_order(ticker, limit_upper_price, volume)
                        print("익절 주문 재설정 : ", ticker, volume)
                        print("Canceled Order: ", upbit.get_order(uuid))
                        print("Make Upper Limit Order : ", limit_upper)

                        if "error" in limit_upper:
                            print("Current Volume : ", upbit.get_balances())
                            volume = upbit.get_balances()[1]["balance"]
                            limit_upper = upbit.sell_limit_order(ticker, limit_upper_price, volume)
                            print("익절 주문 재설정 : ", ticker, volume)
                            print("Canceled Order: ", upbit.get_order(uuid))
                            # print(market)
                            print("Make Upper Limit Order : ", limit_upper)

                        limit_order_uuid.append(limit_upper["uuid"])

                        limit_order_lower_price.append(limit_lower_price)
                        limit_order_upper_price.append(limit_upper_price)
                        limit_order_volume.append(volume)

                        # limit_order_time.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        limit_order_time.append(order_time)

                time.sleep(1)

            # INFO configuration
            print("Order List")
            print(limit_order_lower_price)
            print(limit_order_upper_price)

            print(limit_order_volume)
            print(limit_order_uuid)
            print(limit_order_time)

            total_log = [limit_order_uuid, limit_order_lower_price, limit_order_upper_price, limit_order_volume,
                         limit_order_time]

            # INFO log file Save
            with open('./upbit log/log.txt', 'wb') as file:
                pickle.dump(total_log, file)

            # time.sleep(55)
            # time check
            current_time = (datetime.datetime.now()-datetime.timedelta(minutes=30)).date().strftime("%Y%m%d")
            if trader_start_time != current_time:
                print("Reload Model")
                break
            else:
                print("Trader Start time : ", trader_start_time)
