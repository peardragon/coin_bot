import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import sklearn
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import mysql_func
from datetime import datetime
import pickle
from sklearn.metrics import recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objects as go
import os


pymysql.install_as_MySQLdb()



def get_sql_data(ticker, start_time, finish_time):
    start_time = str(datetime.strptime(str(start_time), '%Y%m%d'))
    finish_time = str(datetime.strptime(str(finish_time), '%Y%m%d'))

    sql = f"SELECT * FROM `coin_min_db`.{ticker} WHERE " \
          f"`coin_min_db`.{ticker}.time BETWEEN '{start_time}' AND '{finish_time}'"

    conn = mysql_func.mysql_conn("coin_min_db")
    cur = conn.cursor()
    cur.execute(sql)
    res_sql = cur.fetchall()

    cur.close()
    conn.close()
    return res_sql

def train_data_processing(sql_data, min_interval, rate_min, render=False):
    X = []

    res_df = pd.DataFrame(sql_data)
    res_arr = res_df.to_numpy()
    res_df.columns = ['time', 'open', 'low', 'high', 'close', 'volume']
    res_df["interval_max"] = res_df.high[::-1].rolling(min_interval).max()[::-1]
    res_df["interval_min"] = res_df.low[::-1].rolling(min_interval).min()[::-1]

    res_df["rate"] = ((res_df.interval_max - res_df.open) / res_df.open) * 100
    res_df["label"] = res_df.rate > rate_min
    y = res_df.label.to_numpy()[:-min_interval]

    data_len = np.shape(res_arr)[0]

    for i in range(data_len - min_interval):
        curr = res_arr[i:i + min_interval]
        curr = curr[:, [1, 3, 5]]
        # make array with 'open' 'high' 'volume' data, and flatten()
        if not render:
            scaled_data = MinMaxScaler().fit_transform(X=curr)
        else:
            scaled_data = curr
        X.append(scaled_data.flatten())

    X = np.asarray(X, dtype=float)

    return X, y


# Define objective functions with own arguments
# study.optimize(lambda trial: objective(trial, X, y), n_trials=100) also possible
class Objective(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)
        boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
        n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=10)
        max_depth = trial.suggest_int('max_depth', 3, 20, step=1)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        clf = LGBMClassifier(boosting_type=boosting_type, n_estimators=n_estimators,
                             max_depth=max_depth, learning_rate=lr, metric="accuracy", verbose=1)

        clf.fit(X_train, y_train)

        score = accuracy_score(y_test, clf.predict(X_test))

        # score = sklearn.model_selection.cross_val_score(clf, X_train, y_train, n_jobs=-1,
        #                                                 verbose=3, scoring="accuracy", cv=3)

        with open(f"./lgbm_model/{trial.number}.pickle", "wb") as model_file:
            pickle.dump(clf, model_file)

        return score


def restore_model_best_params(best_trial_number):
    with open(f"./lgbm_model/{best_trial_number}.pickle", "rb") as fin:
        best_clf = pickle.load(fin)
    return best_clf


def restore_model(filename):
    with open(filename, 'rb') as model:
        loaded_model = pickle.load(model)
    return loaded_model


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error Occured")


## Configs ## #
if __name__ == "__main__":
    start = "20201010"
    finish = "20201020"
    interval = 60
    rate = 1

    # ## Configs ## #
    date_path = start + "_" + finish + "/"
    directory_path = "./lgbm_model/" + date_path
    configs = "_interval_" + str(interval) + "_rate_" + str(rate)
    model_name = "optimized_model" + configs + ".sav"
    file_path = "./lgbm_model/" + date_path + model_name

    if input(" 1. Optimizing Model / 2. Restore Saved Model") == "1":
        res = get_sql_data("btckrw", start, finish)

        ticker_data = pd.DataFrame(res, columns=['time', 'open', 'low', 'high', 'close', 'volume'])
        ticker_data["interval_max"] = ticker_data.high[::-1].rolling(interval).max()[::-1]

        X, y = train_data_processing(res, interval, rate)
        X_origin, y_origin = train_data_processing(res, interval, rate, render=False)

        study = optuna.create_study(direction="maximize")
        study.optimize(Objective(X, y), n_trials=100, timeout=15)

        best_clf_ = restore_model_best_params(study.best_trial.number)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        best_clf_.fit(X_train, y_train, verbose=2)

        with open(file_path, 'wb') as file:
            pickle.dump(best_clf_, file)

        # pred_proba = best_clf_.predict_proba(X_test)
        # pred = [i > 0.8 for i in pred_proba[:, 1]]
        #TODO 뭔가 학습이 잘 안된거 같음. 실제로 라벨과 비교 필요 - y_test 와 비교
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        test_pred = best_clf_.predict(X_test)
        train_pred = best_clf_.predict(X_train)

        # View about Train Data - plotly rendering
        close_train = X_origin[:, -2]

        idx_train_origin = np.argwhere(np.array(y_train) == 1).flatten()
        idx_train_pred = np.argwhere(np.array(train_pred) == 1).flatten()

        idx_test_origin = np.argwhere(np.array(y_test) == 1).flatten()+ len(X_train)
        idx_test_pred = np.argwhere(np.array(test_pred) == 1).flatten() + len(X_train)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_train_origin],
                                 y=ticker_data["open"].iloc[idx_train_origin],
                                 mode="markers", marker=dict(color="black", size=10), opacity=0.7, name="train label"))

        fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_test_origin],
                                 y=ticker_data["open"].iloc[idx_test_origin],
                                 mode="markers", marker=dict(color="black", size=10), opacity=0.7, name="test label"))

        fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_train_pred],
                                 y=ticker_data["open"].iloc[idx_train_pred],
                                 mode="markers", marker=dict(color="red", size=12), opacity=0.7, name="train decision"))

        fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_test_pred],
                                 y=ticker_data["open"].iloc[idx_test_pred],
                                 mode="markers", marker=dict(color="blue", size=12), opacity=0.7, name="test decision"))

        fig.add_trace(go.Scatter(x=ticker_data["time"][:len(close_train)],
                                 y=ticker_data["open"][:len(close_train)],
                                 mode="lines", marker=dict(color="black"), opacity=0.3, name="ticker data"))

        fig.add_trace(go.Scatter(x=ticker_data["time"][:len(close_train)],
                                 y=ticker_data["interval_max"][:len(close_train)],
                                 mode='lines', marker=dict(color="blue"), opacity=0.5, name=f"interval max"))

        fig.update_layout(xaxis_rangeslider_visible=True)
        fig.write_html("./lightgbm_results_train.html")

        recall_score(test_pred, y_test)
    else:
        start_eval = "20201021"
        finish_eval = "20201024"

        print(f"load model fit by data : {start_eval} ~ {finish_eval}")
        res = get_sql_data("btckrw", start_eval, finish_eval)

        ticker_data = pd.DataFrame(res, columns=['time', 'open', 'low', 'high', 'close', 'volume'])
        ticker_data["interval_max"] = ticker_data.high[::-1].rolling(interval).max()[::-1]

        X, y = train_data_processing(res, interval, rate)
        X_origin, y_origin = train_data_processing(res, interval, rate, render=True)

        best_clf_ = restore_model(filename=file_path)

        test_pred = best_clf_.predict(X)

        # plotly rendering
        close = X_origin[:, -2]

        idx_origin = np.argwhere(np.array(test_pred) == 1).flatten()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_origin],
                                 y=ticker_data["open"].iloc[idx_origin],
                                 mode="markers", marker=dict(color="red", size=12), opacity=0.7, name="decision"))

        fig.add_trace(go.Scatter(x=ticker_data["time"],
                                 y=ticker_data["open"],
                                 mode="lines", marker=dict(color="black"), opacity=0.3, name="ticker"))

        fig.add_trace(go.Scatter(x=ticker_data["time"],
                                 y=ticker_data["interval_max"],
                                 mode='lines', marker=dict(color="blue"), opacity=0.5, name=f"interval max"))

        fig.update_layout(xaxis_rangeslider_visible=True)
        fig.write_html("./lightgbm_results_pred.html")
