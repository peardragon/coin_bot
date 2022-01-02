from lightgbm_model import get_sql_data, train_data_processing, \
    train_test_split, LGBMClassifier, timedelta
import sklearn
import numpy as np
import datetime
from sklearn.metrics import precision_score, recall_score
from ray import tune
import ray
import glob
import pickle
from ray.tune.suggest.optuna import OptunaSearch

def restore_optimized_model(filename):
    for file in glob.glob(filename):
        with open(file, 'rb') as model:
            print("restore model in : ", file)
            rate = float(file[file.find("rate_") + 5:file.find(".sav")])
            interval = float(file[file.find("interval_") + 9:file.find("_rate")])
            loaded_model = pickle.load(model)
            break
    return loaded_model, interval, rate


config = {'boosting_type': tune.choice(['gbdt', 'dart']),
          'days': tune.qrandint(1, 14, 1),
          'interval': tune.qrandint(60, 180, 10),
          'rate': tune.uniform(0.3, 0.4),
          'n_estimators': tune.qrandint(100, 2000, 10),
          'max_depth': tune.qrandint(3, 15, 1),
          'lr': tune.loguniform(1e-5, 1e-1),
          'num_leaves': tune.qrandint(2, 2 ** 4),
          'reg_lambda': tune.uniform(0, 0.4)}


def training_function(config, checkpoint_dir=None):
    ticker_config = "btckrw"
    fin_time_auto = config["fin_time"]
    train_interval = config["days"]
    start_time_auto = fin_time_auto - timedelta(days=train_interval)
    fin_time_auto = fin_time_auto.strftime("%Y%m%d")
    start_time_auto = start_time_auto.strftime("%Y%m%d")
    interval = config["interval"]
    rate = config["rate"]
    res = get_sql_data(ticker_config, start_time_auto, fin_time_auto)

    X, y = train_data_processing(res, interval, rate)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    boosting_type = 'gbdt'

    n_estimators = config["n_estimators"]
    max_depth = config["max_depth"]
    num_leaves = config["num_leaves"]
    lr = config["lr"]
    reg_lambda = config["reg_lambda"]

    clf = LGBMClassifier(boosting_type=boosting_type, n_estimators=n_estimators,
                         max_depth=max_depth, learning_rate=lr, reg_lambda=reg_lambda,
                         metric="AUC", verbose=-1, force_col_wise=True, n_jobs=1)
    clf.fit(X_train, y_train, verbose=2)

    score = sklearn.model_selection.cross_val_score(clf, X_test, y_test, n_jobs=1,
                                                    verbose=1, scoring="roc_auc", cv=3)
    score = np.mean(score)
    print(score)
    tune.report(mean_auc=score, done=True)


config["fin_time"] = datetime.datetime.now().date() - timedelta(days=3)
fin_time_auto = (datetime.datetime.now().date() - timedelta(days=3)).strftime("%Y%m%d")

if __name__ == "__main__":
    ray.init(local_mode=True, num_cpus=8, num_gpus=1)
    arg = input(" 1. Optimizing Model / 2. Eval Model : ")

    if arg == "1":
        results = tune.run(training_function,
                           search_alg=OptunaSearch(),
                           mode="max", metric="mean_auc", config=config,
                           num_samples=100, time_budget_s=100,
                           resources_per_trial={"cpu": 2, "gpu": 1 / 8}, verbose=2)

        best_config = results.best_config
        interval = best_config["interval"]
        rate = best_config["rate"]
        print(best_config)

        ticker_config = "btckrw"
        fin_time_auto = best_config["fin_time"]
        train_interval = best_config["days"]
        start_time_auto = fin_time_auto - timedelta(days=train_interval)
        fin_time_auto = fin_time_auto.strftime("%Y%m%d")
        start_time_auto = start_time_auto.strftime("%Y%m%d")
        res = get_sql_data(ticker_config, start_time_auto, fin_time_auto)
        X, y = train_data_processing(res, interval, rate)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf = LGBMClassifier(boosting_type=best_config["boosting_type"], n_estimators=best_config["n_estimators"],
                             max_depth=best_config["max_depth"], learning_rate=best_config["lr"],
                             metric="AUC", verbose=-1, force_col_wise=True, n_jobs=8)

        clf.fit(X_train, y_train, verbose=2)

        test_pred = clf.predict(X_test)
        train_pred = clf.predict(X_train)

        print(recall_score(y_test, test_pred))
        print(recall_score(y_train, train_pred))

        print(precision_score(y_test, test_pred))
        print(precision_score(y_train, train_pred))

        # ## Configs ## #
        date_path = start_time_auto + "_" + fin_time_auto + "/"
        directory_path = "./lgbm_model/ray/" + date_path
        configs = "_interval_" + str(interval) + "_rate_" + str(rate)
        model_name = "optimized_model" + configs + ".sav"
        file_path = "./lgbm_model/ray/" + date_path + model_name

        from lightgbm_model import createFolder
        createFolder(directory_path)

        with open(file_path, 'wb') as file:
            pickle.dump(clf, file)

    else:
        start_eval = "20210905"
        finish_eval = "20210906"

        model, interval, rate = restore_optimized_model(f'./lgbm_model/ray/*_{fin_time_auto}/'
                                                        f'optimized_model_interval_*_rate_*.sav')

        res = get_sql_data("btckrw", start_eval, finish_eval)
        X, y = train_data_processing(res, int(interval), float(rate))
        pred = model.predict(X)

        print(pred)
        print(recall_score(y, pred))
        print(precision_score(y, pred))
