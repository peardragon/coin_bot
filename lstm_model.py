import numpy as np
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, ConvLSTM2D, BatchNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split

# https://keras.io/ko/getting-started/sequential-model-guide/
# https://pozalabs.github.io/lstm/
from lightgbm_model import get_sql_data, train_data_processing, rate_config
import pandas as pd


def f1_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    tp = K.sum(K.cast(y_true * y_pred, 'float32'), axis=0)
    tn = K.sum(K.cast((1.0 - y_true) * (1.0 - y_pred), 'float32'), axis=0)
    fp = K.sum(K.cast((1.0 - y_true) * y_pred, 'float32'), axis=0)
    fn = K.sum(K.cast(y_true * (1.0 - y_pred), 'float32'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def lstm_input_processing(res_df, interval):
    arr = res_df.iloc[:,[1,2,3,4,5]].to_numpy()
    X = []
    for i in range(arr.shape[0] - interval):
        curr = arr[i:i + interval]
        scaled_data = MinMaxScaler().fit_transform(X=curr)
        X.append(scaled_data)

    X = np.asarray(X)
    return X

def lstm_train_data_processing(sql_data, min_interval, rate_min, render=False):
    X = []

    res_df = pd.DataFrame(sql_data)
    res_arr = res_df.to_numpy()
    #
    res_df.columns = ['time', 'open', 'low', 'high', 'close', 'volume']
    res_df["interval_max"] = res_df.high[::-1].rolling(min_interval).max()[::-1]
    res_df["interval_min"] = res_df.low[::-1].rolling(min_interval).min()[::-1]

    # res_df["rate"] = ((res_df.interval_max - res_df.open) / res_df.open) * 100
    res_df["rate"] = ((res_df.interval_max - res_df.close) / res_df.close) * 100
    res_df["label"] = res_df.rate > rate_min
    res_df["label"] = res_df["label"].astype(int)

    data_len = np.shape(res_arr)[0]

    for i in range(data_len - min_interval):
        curr = res_arr[i:i + min_interval]
        curr = curr[:, [1,2,3,4,5]]
        if not render:
            scaled_data = MinMaxScaler().fit_transform(X=curr)
        else:  # if render
            scaled_data = curr
        X.append(scaled_data)

    X = np.asarray(X, dtype=float)
    # [0:interval] 까지의 data 를 통해 interval 열 의 label 을 y 로
    y = res_df.label.to_numpy()[min_interval:].reshape(-1,1)
    print("Shape : ", (X.shape, y.shape))
    print(X[0],y[0])

    return X, y

start = "20210820"
finish = "20210825"

interval = 60
# automation rate configuration
print("rate config .. ")
res = get_sql_data("btckrw", start, finish)
rate = rate_config(res, interval)
res_df = pd.DataFrame(res)

X, y = lstm_train_data_processing(res, interval, rate)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#


def model(interval):
    batch_size = 60
    # https://tykimos.github.io/2017/04/09/RNN_Getting_Started/
    timesteps = interval
    data_dim = 5 # open close low high volume

    model = Sequential()
    # LSTM 레이어는 return_sequences 인자에 따라 마지막 시퀀스에서 한 번만 출력할 수 있고 각 시퀀스에서 출력을 할 수 있습니다.
    # many to many 문제를 풀거나 LSTM 레이어를 여러개로 쌓아올릴 때는 return_sequence=True 옵션을 사용합니다.

    # state : 도출된 현재 상태의 가중치가 다음 샘플 학습 시의 초기 상태로 입력됨을 알 수 있습니다.
    # model.add(LSTM(200, return_sequences=True, input_shape=(timesteps, data_dim)))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    # model.add(LSTM(100, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(50, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(25))
    # model.add(Dropout(0.2))
    model.add(LSTM(200, input_shape=(timesteps, data_dim)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss= lambda y_true, y_pred: binary_crossentropy(y_true, y_pred, from_logits=True),
                  optimizer=Adam(amsgrad=True),
                  metrics=['AUC', 'Recall', "Precision"])

    return model


def batch_reshape(arr, interval):
    return arr[:int(arr.shape[0]/interval)*interval]


# Generate dummy training data
model = model(interval)
print(model.summary())
X_train = batch_reshape(X_train, interval)
y_train = batch_reshape(y_train, interval)
X_test = batch_reshape(X_test, interval)
y_test = batch_reshape(y_test, interval)

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64,
          epochs=1500, shuffle=True)

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history[list(hist.history.keys())[0]], 'y', label='train loss')
loss_ax.plot(hist.history[list(hist.history.keys())[4]], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')
print(hist.history.keys())

acc_ax.plot(hist.history[list(hist.history.keys())[1]], "g", alpha=0.5, label='train auc')
acc_ax.plot(hist.history[list(hist.history.keys())[2]], "g", alpha=0.8, label='train recall')
acc_ax.plot(hist.history[list(hist.history.keys())[3]], "g", alpha=1,  label='train precision')

acc_ax.plot(hist.history[list(hist.history.keys())[5]], "b", alpha=0.5,  label='val auc')
acc_ax.plot(hist.history[list(hist.history.keys())[6]], "b", alpha=0.8,  label='val recall')
acc_ax.plot(hist.history[list(hist.history.keys())[7]], "b", alpha=1,  label='val precision')
acc_ax.set_ylabel('metrics')
acc_ax.legend(loc='upper right')

plt.show()

# pred = model.predict(X_test)
# pred = model.predict(X_test, batch_size=300)
score = model.evaluate(X_test, y_test)

print(score)