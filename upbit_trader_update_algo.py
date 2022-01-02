import time

from upbit_trader import restore_optimized_model
import datetime
from lightgbm_model import CoinObjective
import pickle
import pandas as pd
from lightgbm_model import train_data_processing, get_sql_data, Objective, \
    restore_model_best_params, train_test_split, createFolder
import optuna
if __name__ == "__main__":
    # Model Connection

    while True:
        fin_time_auto = datetime.datetime.now().date()
        fin_time_auto = fin_time_auto.strftime("%Y%m%d")
        print("Current Time : ", fin_time_auto)
        trader_start_time = datetime.datetime.now().date().strftime("%Y%m%d")
        while True:
            try:
                model, interval, rate = restore_optimized_model(f'./lgbm_model/*_{fin_time_auto}/'
                                                                f'optimized_model_interval_*_rate_*.sav')
            except Exception as e:
                print(e)
                print("Model un-Exist. Make model")

                study = optuna.create_study(direction="maximize")
                fin_time = datetime.datetime.now().date()
                study.optimize(CoinObjective(fin_time), n_trials=100, timeout=1500)
                best_clf_ = restore_model_best_params(study.best_trial.number)

                days = study.best_params['days']
                interval = study.best_params['interval']
                rate = round(study.best_params['rate'], 3)

                # config update #
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
                # config update #

                X, y = train_data_processing(res, interval, rate)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

                best_clf_.fit(X_train, y_train, verbose=2)

                print("Optimize and Fit Finish : Save model")

                createFolder(directory_path)

                with open(file_path, 'wb') as file:
                    pickle.dump(best_clf_, file)

            current_time = datetime.datetime.now().date().strftime("%Y%m%d")
            if trader_start_time != current_time:
                print(trader_start_time, current_time)
                print("Re Construct Model")
                break
            else:
                print("Current Model Already Constructed", end="\r")

            for i in range(60):
                print("."*(i+1), end="\r")
                time.sleep(0.5)
            print("")
            time.sleep(30)
