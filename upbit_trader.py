from pyupbit import WebSocketManager
import pyupbit
# from min_algorithm import restore_model
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import datetime
import private_key as key

# ## Default Algorithm Config ## #
fin_time_auto = datetime.datetime.now().date()
start_time_auto = fin_time_auto - datetime.timedelta(days=5)

fin_time_auto = fin_time_auto.strftime("%Y%m%d")
start_time_auto = start_time_auto.strftime("%Y%m%d")

time_config = start_time_auto + "_" + fin_time_auto
interval_config = 60
rate_config = 0.55
ticker_config = "btckrw"


def restore_model(filename):
    with open(filename, 'rb') as model:
        loaded_model = pickle.load(model)
    return loaded_model


if __name__ == "__main__":
    # wm = WebSocketManager("ticker", ["KRW-BTC"])
    # data = wm.get()
    # print(data)
    # # wm.terminate()
    upbit = pyupbit.Upbit(key.upbit_access, key.upbit_secret)
    # custom
    # config = dict(time="20201010_20201020", interval=80, rate=0.7)
    config = dict(time=time_config, interval=interval_config, rate=rate_config)

    try:
        model = restore_model(f'./lgbm_model/{config["time"]}/'
                              f'optimized_model_interval_{str(config["interval"])}_rate_{str(config["rate"])}.sav')
    except:
        print("Model un-Exist. Make model")
        import pandas as pd
        from lightgbm_model import train_data_processing, get_sql_data, Objective, \
            restore_model_best_params, train_test_split, createFolder
        import optuna

        start = start_time_auto
        finish = fin_time_auto
        interval = interval_config
        rate = rate_config
        res = get_sql_data(ticker_config, start_time_auto, fin_time_auto)

        date_path = start + "_" + finish + "/"
        directory_path = "./lgbm_model/" + date_path
        configs = "_interval_" + str(interval) + "_rate_" + str(rate)
        model_name = "optimized_model" + configs + ".sav"
        file_path = "./lgbm_model/" + date_path + model_name

        ticker_data = pd.DataFrame(res, columns=['time', 'open', 'low', 'high', 'close', 'volume'])
        ticker_data["interval_max"] = ticker_data.high[::-1].rolling(interval).max()[::-1]
        ticker_data = ticker_data[interval:].reset_index(drop=True)

        X, y = train_data_processing(res, interval, rate)
        X_origin, y_origin = train_data_processing(res, interval, rate, render=False)

        study = optuna.create_study(direction="maximize")
        study.optimize(Objective(X, y), n_trials=100, timeout=600)

        best_clf_ = restore_model_best_params(study.best_trial.number)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        best_clf_.fit(X_train, y_train, verbose=2)

        print("Optimize and Fit Finish : Save model")

        createFolder(directory_path)

        with open(file_path, 'wb') as file:
            pickle.dump(best_clf_, file)

        model = restore_model(f'./lgbm_model/{config["time"]}/'
                              f'optimized_model_interval_{str(config["interval"])}_rate_{str(config["rate"])}.sav')

    print("Algorithm Connect")

    ticker = "KRW-BTC"
    limit_upper = limit_lower = None

    limit_order_uuid = []
    limit_order_lower_price = []
    limit_order_volume = []
    limit_order_time = []

    order_dict = upbit.get_order(ticker, state="wait")
    for i in order_dict:
        if i["ord_type"] == "limit":
            limit_order_uuid.append(i["uuid"])
            lower_price = round(round(
                round(float(i["price"]) / (1 + config["rate"] / 100), -3) * (1 - config["rate"] / 100), -3), 1)
            limit_order_lower_price.append(lower_price)
            limit_order_volume.append(i["remaining_volume"])
            limit_order_time.append(i["created_at"])

    while True:
        X = []

        try:
            df = pyupbit.get_ohlcv("KRW-BTC", count=config["interval"] + 1, interval="minute1")
            data_arr = df[["open", "high", "volume"]].iloc[:-1].to_numpy()
            scaled_data = MinMaxScaler().fit_transform(X=data_arr)
            X.append(scaled_data.flatten())
            X = np.asarray(X, dtype=float)
        except TypeError:
            continue

        pred = model.predict(X)
        print("Current Time: ", datetime.datetime.now())
        print("Decision: ", pred[0])
        if pred[0]:
            # 매수 금액은 수수료를 제외한 금액입니다
            current_price = pyupbit.get_current_price(ticker)
            order = upbit.buy_market_order(ticker, 10000)
            print(order)
            print("Current Price: ", current_price)

            if "error" in order:
                pass
            else:
                while True:
                    # if len(upbit.get_order(ticker)) == 0:
                    if upbit.get_order(order["uuid"])["state"] != "wait":
                        volume = upbit.get_order(order["uuid"])["executed_volume"]
                        limit_upper_price = round(current_price * (1 + config["rate"] / 100), -3)
                        limit_lower_price = round(current_price * (1 - config["rate"] / 100), -3)
                        limit_upper = upbit.sell_limit_order(ticker, limit_upper_price, volume)
                        print(limit_upper)

                        limit_order_lower_price.append(limit_lower_price)
                        limit_order_uuid.append(limit_upper["uuid"])
                        limit_order_volume.append(volume)
                        limit_order_time.append(limit_upper["created_at"])
                        break

        # 손절 확인
        current_price = pyupbit.get_current_price(ticker)
        for price in limit_order_lower_price:
            if price >= current_price:
                idx = limit_order_lower_price.index(price)
                limit_order_lower_price.remove(price)

                uuid = limit_order_uuid[idx]
                volume = limit_order_volume[idx]
                time = limit_order_time[idx]

                limit_order_uuid.remove(uuid)
                limit_order_volume.remove(volume)
                limit_order_time.remove(time)

                upbit.cancel_order(uuid)
                market = upbit.sell_market_order(ticker, volume)
                print("손절")
                print(market)
                while True:
                    if upbit.get_order(market["uuid"])["state"] != "wait":
                        break

        # 손절 이후 order list Synchronize

        complement = [i for i in [j for j in limit_order_uuid]
                      if i not in [k["uuid"] for k in upbit.get_order(ticker)]]

        if len(complement) != 0:
            for item in complement:
                idx = limit_order_uuid.index(item)

                uuid = limit_order_uuid[idx]
                volume = limit_order_volume[idx]
                current_price = limit_order_lower_price[idx]
                time = limit_order_time[idx]

                limit_order_uuid.remove(uuid)
                limit_order_volume.remove(volume)
                limit_order_lower_price.remove(current_price)
                limit_order_time.remove(time)

        print("Order List")
        print(limit_order_lower_price)
        print(limit_order_volume)
        print(limit_order_uuid)
        print(limit_order_time)
        # print(df)
        # print(data_arr)
        time.sleep(50)

    # upbit.sell_limit_order("KRW-XRP", 600, 20) - current price buy 하고, executed 되면 limit order 걸기
