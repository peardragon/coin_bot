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
from sklearn.metrics import recall_score

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


def train_data_processing(sql_data, interval, rate_min, render=False):
    _X = []

    res_df = pd.DataFrame(sql_data)
    res_arr = res_df.to_numpy()
    res_df.columns = ['time', 'open', 'low', 'high', 'close', 'volume']
    res_df["interval_max"] = res_df.high[::-1].rolling(interval).max()[::-1]
    res_df["interval_min"] = res_df.low[::-1].rolling(interval).min()[::-1]

    res_df["rate"] = ((res_df.interval_max - res_df.open) / res_df.open) * 100
    res_df["label"] = res_df.rate > rate_min
    _y = res_df.label.to_numpy()[:-interval]

    data_len = np.shape(res_arr)[0]

    for i in range(data_len - interval):
        curr = res_arr[i:i + interval]
        # make array with 'open' 'high' 'volume' data, and flatten()
        _X.append(curr[:, [1, 3, 5]].flatten())

    _X = np.asarray(_X, dtype=float)

    if render:
        plt.plot(res_df.open)
        idx = np.argwhere(res_df.label.to_numpy() > 0.5)
        plt.scatter(idx, res_df.open.to_numpy()[idx], c="red")
        plt.show()

    return _X, _y


def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=10)
    max_depth = trial.suggest_int('max_depth', 3, 20, step=1)
    clf = LGBMClassifier(boosting_type=boosting_type, n_estimators=n_estimators,
                         max_depth=max_depth, metric="accuracy")

    score = sklearn.model_selection.cross_val_score(clf, X_train, y_train, n_jobs=-1,
                                                    verbose=3, scoring="accuracy", cv=3)
    accuracy = score.mean()

    with open(f"./lgbm_model/{trial.number}.pickle", "wb") as model_file:
        pickle.dump(clf, model_file)

    return accuracy


def restore_model_best_params(best_trial_number):
    with open(f"./lgbm_model/{best_trial_number}.pickle", "rb") as fin:
        best_clf = pickle.load(fin)
    return best_clf

def restore_model(filename):
    with open(filename, 'rb') as model:
        loaded_model = pickle.load(model)
    return loaded_model


start = "20200801"
finish = "20200811"
file_name = "./lgbm_model/optimized_model.sav"

if input(" 1. Optimizing Model / 2. Restore Saved Model") == "1":
    res = get_sql_data("btckrw", start, finish)
    X, y = train_data_processing(res, 60, 1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_clf_ = restore_model_best_params(study.best_trial.number)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    best_clf_.fit(X_train, y_train, verbose=2)

    with open(file_name, 'wb') as file:
        pickle.dump(best_clf_, file)

    pred_proba = best_clf_.predict_proba(X_test)
    pred =[i > 0.8 for i in pred_proba[:, 1]]

    recall_score(pred, y_test)
else:
    print(f"load model fit by data : {start} ~ {finish}")
    res = get_sql_data("btckrw", start, finish)
    X, y = train_data_processing(res, 60, 1)

    best_clf_ = restore_model(filename=file_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    # best_clf_.fit(X_train, y_train, verbose=2)
    pred_proba = best_clf_.predict_proba(X_test)
    pred = [i > 0.8 for i in pred_proba[:, 1]]
    close = X_test[:, -2]

    plt.plot(range(len(close)), close, c="k", alpha=0.3)
    idx = np.argwhere(np.array(pred)==True)
    plt.scatter(idx, close[idx], s=7, c="r", alpha=0.5)
    plt.show()

    recall_score(pred, y_test)

# def decision(self, data):
#     data = [[i[0], i[2], i[4]] for i in data]
#     X = np.asarray(data, dtype=float)
#     pred_proba = best_clf_.predict_proba(X_test)
#     pred = [i > 0.8 for i in pred_proba[:, 1]]
#
#     if pred.item() is True:
#         decison = 'buy'
#     else:
#         decision = "stay"
#     # open low high close volume
#     current_price = data[-1, 3]
#     limit_price_lower = current_price * 0.99
#     limit_price_upper = current_price * 1.01
#
#     return {'decision': decision, 'limit_high': limit_price_upper, 'limit_low': limit_price_lower}