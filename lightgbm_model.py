import pymysql
import pandas as pd
import numpy as np
from min_craw_2 import Collector
import optuna
import sklearn
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import mysql_func
from datetime import datetime, timedelta
import pickle
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import time
import plotly.graph_objects as go
import os
import glob

pymysql.install_as_MySQLdb()


# fee = 0.05%
def restore_optimized_model(filename):
    for file in glob.glob(filename):
        with open(file, 'rb') as model:
            print("restore model in : ", file)
            rate = float(file[file.find("rate_") + 5:file.find(".sav")])
            interval = float(file[file.find("interval_") + 9:file.find("_rate")])
            loaded_model = pickle.load(model)
            break
    return loaded_model, interval, rate


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


def rate_config(sql_data, interval):
    res_df = pd.DataFrame(sql_data)
    res_df.columns = ['time', 'open', 'low', 'high', 'close', 'volume']
    res_df = res_df.set_index("time", drop=True)
    res_df.index.astype('datetime64[ns]')
    res_df_interval = res_df.resample(f"{interval}T").agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    res_df_interval["rate"] = (res_df_interval.high - res_df_interval.open) / res_df_interval.open * 100
    rate_upper = res_df_interval.rate.mean()
    rate_upper = round(rate_upper, 3)
    res_df_interval["rate"] = (res_df_interval.low - res_df_interval.open) / res_df_interval.open * 100
    rate_lower = res_df_interval.rate.mean()
    print("Automation Rate config : ", rate_upper)

    return rate_upper


def train_data_processing_1(data):
    X = []
    # curr = data[:, [0, 2, 4]]
    curr = data[:, [0, 1, 2, 3, 4]]
    scaled_data = MinMaxScaler().fit_transform(X=curr)
    X.append(scaled_data.flatten())
    X = np.asarray(X, dtype=float)
    return X


def train_data_processing(sql_data, min_interval, rate_min, render=False):
    X = []

    res_df = pd.DataFrame(sql_data)
    res_arr = res_df.to_numpy()
    res_df.columns = ['time', 'open', 'low', 'high', 'close', 'volume']
    res_df["interval_max"] = res_df.high[::-1].rolling(min_interval).max()[::-1]
    res_df["interval_min"] = res_df.low[::-1].rolling(min_interval).min()[::-1]

    # res_df["rate"] = ((res_df.interval_max - res_df.open) / res_df.open) * 100
    res_df["rate"] = ((res_df.interval_max - res_df.close) / res_df.close) * 100
    res_df["rate_low"] = ((res_df.interval_min - res_df.close) / res_df.close) * 100
    res_df["loss_cut"] = res_df.rate_low > -rate_min * 1.5
    res_df["profit_cut"] = res_df.rate > rate_min
    # res_df["label"] = res_df.loss_cut * res_df.profit_cut

    def cond(df):
        if df["loss_cut"] is False:
            return -1
        elif df["profit_cut"] is True:
            return 1
        else:
            return 0

    res_df["label"] = res_df.apply(cond, axis=1)

    data_len = np.shape(res_arr)[0]

    for i in range(data_len - min_interval):
        curr = res_arr[i:i + min_interval]
        # curr = curr[:, [1, 3, 5]]
        curr = curr[:, [1, 2, 3, 4, 5]]
        # make array with 'open' 'high' 'volume' data, and flatten()
        if not render:
            scaled_data = MinMaxScaler().fit_transform(X=curr)
        else:  # if render
            scaled_data = curr
        X.append(scaled_data.flatten())

    X = np.asarray(X, dtype=float)
    # [0:interval] 까지의 data 를 통해 interval 열 의 label 을 y 로
    y = res_df.label.to_numpy()[min_interval:]

    return X, y


def target_broad(y, interval):
    if interval > 0:
        temp = np.roll(y, interval)
        temp[:interval] = np.zeros(interval)
        y_broad = y | temp
    elif interval < 0:
        temp = np.roll(y, interval)
        temp[len(temp) + interval:] = np.zeros(-interval)
        y_broad = y | temp
    else:
        y_broad = y
    return y_broad

# Define objective functions with own arguments
# study.optimize(lambda trial: objective(trial, X, y), n_trials=100) also possible
class Objective(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)
        boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
        n_estimators = trial.suggest_int('n_estimators', 100, 2000, step=10)
        max_depth = trial.suggest_int('max_depth', 3, 15, step=1)
        num_leaves = trial.suggest_int('num_leaves', 2, 2 ** int(max_depth / 2), log=True)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        clf = LGBMClassifier(boosting_type=boosting_type, n_estimators=n_estimators,
                             max_depth=max_depth, learning_rate=lr, num_leaves=num_leaves,
                             metric="AUC", verbose=1, force_col_wise=True,
                             num_threads=8)

        clf.fit(X_train, y_train)

        # score = accuracy_score(y_test, clf.predict(X_test))
        # score = precision_score(y_test, clf.predict(X_test), zero_division=0)
        # score = recall_score(y_test, clf.predict(X_test), zero_division=0)

        score = sklearn.model_selection.cross_val_score(clf, X_test, y_test, n_jobs=-1,
                                                        verbose=3, scoring="roc_auc", cv=3)
        score = np.mean(score)
        print("Precision Score : ", score)

        with open(f"./lgbm_model/{trial.number}.pickle", "wb") as model_file:
            pickle.dump(clf, model_file)

        return score


class CoinObjective(object):
    def __init__(self, fin_time):
        # Input : datetime.datetime.now().date()
        self.fin_time = fin_time
        self.ticker = "btckrw"

    def __call__(self, trial):
        ticker_config = self.ticker
        fin_time_auto = self.fin_time
        train_interval = trial.suggest_int("days", 3, 10, step=1)
        start_time_auto = fin_time_auto - timedelta(days=train_interval)
        fin_time_auto = fin_time_auto.strftime("%Y%m%d")
        start_time_auto = start_time_auto.strftime("%Y%m%d")
        interval = trial.suggest_int("interval", 30, 90, step=10)
        rate = trial.suggest_float("rate", 0.2, 0.5)
        res = get_sql_data(ticker_config, start_time_auto, fin_time_auto)
        reg_lambda = trial.suggest_float("reg_lambda", 0, 0.4)

        X, y = train_data_processing(res, interval, rate)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        boosting_type = 'dart'

        n_estimators = trial.suggest_int('n_estimators', 100, 2000, step=10)
        max_depth = trial.suggest_int('max_depth', 3, 15, step=1)
        num_leaves = trial.suggest_int('num_leaves', 2, 15, log=True)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        clf = LGBMClassifier(boosting_type=boosting_type, n_estimators=n_estimators,
                             max_depth=max_depth, learning_rate=lr, num_leaves=num_leaves,
                             reg_lambda=reg_lambda,
                             # metric="AUC",
                             metric="multi_logloss",
                             verbose=0, force_col_wise=True,
                             n_jobs=1)

        clf.fit(X_train, y_train)
        # np.asarray([np.zeros(1) + y_train[1:]], dtype=bool)
        # score = accuracy_score(y_test, clf.predict(X_test))
        # score = precision_score(y_test, clf.predict(X_test), zero_division=0)
        # score = recall_score(y_test, clf.predict(X_test), zero_division=0)

        # score = sklearn.model_selection.cross_val_score(clf, X_test, y_test, n_jobs=-1,
        #                                                 verbose=3, scoring="roc_auc", cv=3)
        score = []
        min_broad = -1
        max_broad = 5
        denominator = 0

        for i in range(min_broad, max_broad):
            # val = sklearn.model_selection.cross_val_score(clf, X_test, target_broad(y_test, -i), n_jobs=-1,
            #                                               verbose=3, scoring="roc_auc", cv=3)
            # val = roc_auc_score(target_broad(y_test, -i), clf.predict(X_test), multi_class='ovo')
            pr = precision_score(target_broad(y_test, -i), clf.predict(X_test), average='weighted', zero_division=0)
            rc = recall_score(target_broad(y_test, -i), clf.predict(X_test), average='weighted', zero_division=0)
            val = pr*0.7 + rc*0.3
            val = val*(max_broad-np.abs(i))
            score.append(np.mean(val))
            denominator += max_broad-np.abs(i)
        score = np.sum(score) / denominator
        print("Score : ", score)

        with open(f"./lgbm_model/{trial.number}.pickle", "wb") as model_file:
            pickle.dump(clf, model_file)

        return score


def precision_recall_score(y, y_pred, precision_weight):
    pr = precision_score(y, y_pred, average='weighted', zero_division=0)
    rc = recall_score(y, y_pred, average='weighted', zero_division=0)
    val = pr*precision_weight + rc*(1-precision_weight)
    return val


def broad_target_score(y, y_pred, metric, min_broad, max_broad):
    score = []
    denominator = 0
    for i in range(min_broad, max_broad):
        val = metric(target_broad(y, -i), y_pred)
        val = val * (max_broad - np.abs(i))
        score.append(np.mean(val))
        denominator += max_broad - np.abs(i)

    score = np.sum(score) / denominator
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


def ticker_df_making(res, interval):
    ticker_data = pd.DataFrame(res, columns=['time', 'open', 'low', 'high', 'close', 'volume'])
    ticker_data["interval_max"] = ticker_data.high[::-1].rolling(interval).max()[::-1]
    ticker_data = ticker_data[interval:].reset_index(drop=True)
    return ticker_data


def rendering_model(save_name, clf, res, interval, rate, X_train, X_test, y_train, y_test, ticker_data):
    X_origin, y_origin = train_data_processing(res, interval, rate, render=True)

    test_pred = clf.predict(X_test)
    train_pred = clf.predict(X_train)

    # View about Train Data - plotly rendering
    close_train = X_origin[:, -2]

    idx_train_origin = np.argwhere(np.array(y_train) == 1).flatten()
    idx_train_pred = np.argwhere(np.array(train_pred) == 1).flatten()

    idx_test_origin = np.argwhere(np.array(y_test) == 1).flatten() + len(X_train)
    idx_test_pred = np.argwhere(np.array(test_pred) == 1).flatten() + len(X_train)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_train_origin],
                             y=ticker_data["close"].iloc[idx_train_origin],
                             mode="markers", marker=dict(color="black", size=10), opacity=0.7, name="train label"))

    fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_test_origin],
                             y=ticker_data["close"].iloc[idx_test_origin],
                             mode="markers", marker=dict(color="black", size=10), opacity=0.7, name="test label"))

    fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_train_pred],
                             y=ticker_data["close"].iloc[idx_train_pred],
                             mode="markers", marker=dict(color="red", size=12), opacity=0.7, name="train decision"))

    fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_test_pred],
                             y=ticker_data["close"].iloc[idx_test_pred],
                             mode="markers", marker=dict(color="blue", size=12), opacity=0.7, name="test decision"))

    fig.add_trace(go.Scatter(x=ticker_data["time"][:len(close_train)],
                             y=ticker_data["close"][:len(close_train)],
                             mode="lines", marker=dict(color="black"), opacity=0.3, name="ticker data"))

    fig.add_trace(go.Scatter(x=ticker_data["time"][:len(close_train)],
                             y=ticker_data["interval_max"][:len(close_train)],
                             mode='lines', marker=dict(color="blue"), opacity=0.5, name=f"interval max"))

    fig.update_layout(xaxis_rangeslider_visible=True)
    fig.show()
    fig.write_html(save_name)

    # print(recall_score(y_test, test_pred))
    # print(recall_score(y_train, train_pred))
    #
    # print(precision_score(y_test, test_pred))
    # print(precision_score(y_train, train_pred))


def render_eval(save_name, ticker_data, y, test_pred):
    idx_origin = np.argwhere(np.array(test_pred) == 1).flatten()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_origin],
                             y=ticker_data["close"].iloc[idx_origin],
                             mode="markers", marker=dict(color="red", size=12), opacity=0.7, name="decision"))

    idx = np.argwhere(np.array(y) == 1).flatten()
    fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx],
                             y=ticker_data["close"].iloc[idx],
                             mode="markers", marker=dict(color="blue", size=12), opacity=0.9, name="True"))

    fig.add_trace(go.Scatter(x=ticker_data["time"],
                             y=ticker_data["close"],
                             mode="lines", marker=dict(color="black"), opacity=0.3, name="ticker"))

    fig.add_trace(go.Scatter(x=ticker_data["time"],
                             y=ticker_data["interval_max"],
                             mode='lines', marker=dict(color="blue"), opacity=0.5, name=f"interval max"))

    fig.update_layout(xaxis_rangeslider_visible=True)
    fig.show()
    fig.write_html(save_name)
    # print(recall_score(y, test_pred))
    # print(precision_score(y, test_pred))


# ## Configs ## #
if __name__ == "__main__":

    start_db = "20170101"
    start_db = datetime.strptime(start_db, '%Y%m%d')
    start_db = start_db.strftime('%Y-%m-%d 00:00:00')
    ticker = "KRW-BTC"
    now = datetime.now().strftime('%Y-%m-%d 00:00:00')
    collector = Collector()
    collector.min_craw_db_ticker(start_db, now, ticker)

    start = "20210829"
    finish = "20210903"
    start_eval = "20210910"
    finish_eval = "20210911"

    fin_time_auto = (datetime.now().date() - timedelta(days=3)).strftime("%Y%m%d")

    arg = input(" 1. Optimizing Model / 2. Eval Model / 3. Optimizing Model (v2) / 4. Eval with input val: ")

    if arg == "1":

        # automation rate configuration
        interval = 60
        print("rate config .. ")
        res = get_sql_data("btckrw", start, finish)
        rate = rate_config(res, interval)

        # ## Default Configs ## #
        date_path = start + "_" + finish + "/"
        directory_path = "./lgbm_model/" + date_path
        configs = "_interval_" + str(interval) + "_rate_" + str(rate)
        model_name = "optimized_model" + configs + ".sav"
        file_path = "./lgbm_model/" + date_path + model_name

        ticker_data = ticker_df_making(res, interval)
        X, y = train_data_processing(res, interval, rate)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        study = optuna.create_study(direction="maximize")
        study.optimize(Objective(X, y), n_trials=500, timeout=600)

        best_clf_ = restore_model_best_params(study.best_trial.number)
        best_clf_.fit(X_train, y_train, verbose=2)

        print("Optimize and Fit Finish : Save model")

        createFolder(directory_path)

        with open(file_path, 'wb') as file:
            pickle.dump(best_clf_, file)

        print("Model Check with train data ")
        rendering_model("./lightgbm_results_train.html", best_clf_, res, interval, rate, X_train, X_test, y_train,
                        y_test, ticker_data)

    elif arg == "2":

        # automation rate configuration
        interval = 60
        print("rate config .. ")
        res = get_sql_data("btckrw", start, finish)
        rate = rate_config(res, interval)

        # ## Default Configs ## #
        date_path = start + "_" + finish + "/"
        directory_path = "./lgbm_model/" + date_path
        configs = "_interval_" + str(interval) + "_rate_" + str(rate)
        model_name = "optimized_model" + configs + ".sav"
        file_path = "./lgbm_model/" + date_path + model_name

        print(f"load model fit by data : {start_eval} ~ {finish_eval}")
        res = get_sql_data("btckrw", start_eval, finish_eval)
        ticker_data = ticker_df_making(res, interval)

        X, y = train_data_processing(res, interval, rate)
        X_origin, y_origin = train_data_processing(res, interval, rate, render=True)

        best_clf_ = restore_model(filename=file_path)
        test_pred = best_clf_.predict(X)

        render_eval("./lightgbm_results_pred.html", ticker_data, y, test_pred)

    elif arg == "3":

        print("Optimizing Model with Advanced objective")

        fin_time = datetime.now().date() - timedelta(days=3)
        fin_time = input("Fin time config : ")
        fin_time = datetime.strptime(str(fin_time), '%Y%m%d')
        start_study = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(CoinObjective(fin_time), n_trials=100, timeout=1000, n_jobs=6)
        print("Optimize Stopped Total time : ", time.time()-start_study)
        best_clf_ = restore_model_best_params(study.best_trial.number)
        days = study.best_params['days']
        interval = study.best_params['interval']
        rate = study.best_params['rate']

        # Update config
        # ## Configs ## #
        start = (fin_time - timedelta(days=days)).strftime("%Y%m%d")
        finish = fin_time.strftime("%Y%m%d")
        date_path = start + "_" + finish + "/"
        directory_path = "./lgbm_model/" + date_path
        configs = "_interval_" + str(interval) + "_rate_" + str(rate)
        model_name = "optimized_model" + configs + ".sav"
        file_path = "./lgbm_model/" + date_path + model_name

        # ## Configs ## #

        res = get_sql_data("btckrw", start, finish)
        ticker_data = ticker_df_making(res, int(interval))

        X, y = train_data_processing(res, int(interval), float(rate))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        best_clf_.fit(X_train, y_train, verbose=2)

        print("Optimize and Fit Finish : Save model")

        createFolder(directory_path)

        with open(file_path, 'wb') as file:
            pickle.dump(best_clf_, file)

        print("Model Check with train data ")
        rendering_model("./lightgbm_results_train_2.html", best_clf_, res, interval, rate,
                        X_train, X_test, y_train, y_test, ticker_data)

        with open(file_path, 'wb') as file:
            pickle.dump(best_clf_, file)

    elif arg == "4":
        start_time = input("20200101 form : ")
        fin_time = input("20201201 form : ")
        interval, rate = None, None

        try:
            model, interval, rate = restore_optimized_model(f'./lgbm_model/{start_time}_{fin_time}/'
                                                            f'optimized_model_interval_*_rate_*.sav')
        except OSError:
            print("Un-Exist Model")
            exit()

        if interval and rate is not None:
            start_eval = input("Start Eval")
            finish_eval = input("Finish Eval")
            res = get_sql_data("btckrw", start_eval, finish_eval)
            ticker_data = ticker_df_making(res, int(interval))
            X, y = train_data_processing(res, int(interval), float(rate))
            pred = model.predict(X)

            print("PR Score : ", precision_recall_score(y, pred, precision_weight=0.7))

            # precision = broad_target_score(y, pred, precision_score, -1, 5)
            # recall = broad_target_score(y, pred, recall_score, -1, 5)

            # print("Precision : ", precision)
            # print("Recall : ", recall)
            #
            # print(recall_score(y, pred))
            # print(precision_score(y, pred))

            render_eval("./lightgbm_results_train_eval2.html", ticker_data, y, pred)




    else:
        pass
