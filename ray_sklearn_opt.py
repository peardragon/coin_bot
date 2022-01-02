from tune_sklearn import TuneSearchCV
from lightgbm_model import get_sql_data, train_data_processing, \
    train_test_split, LGBMClassifier, timedelta
# Other imports
import scipy
from ray import tune
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris

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