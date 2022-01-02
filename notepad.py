# # import pickle
# #
# # with open('./upbit log/log.txt', 'rb') as file:
# #     total_log = pickle.load(file)
# #
# # limit_order_lower_price = total_log[0]
# # limit_order_volume = total_log[1]
# # limit_order_uuid = total_log[2]
# # limit_order_time = total_log[3]
#
# # import datetime
# #
# # trader_start_time = datetime.datetime.now().date().strftime("%Y%m%d")
# # print(trader_start_time)
#
# import datetime
# import os, sys, time
#
# def main():
#     while True:
#         i = 0
#         start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         print(start_time)
#         while True:
#             i += 1
#             print(i)
#             time.sleep(0.3)
#             current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             print(current_time)
#             if start_time != current_time:
#                 break
#
#
#
# main()
# print("1")

import pyupbit
import private_key as key
import datetime
# order = pyupbit.get_orderbook("KRW-BTC")
# for i in order[0]["orderbook_units"]:
#     print(i["ask_price"], end="\r") # 매도가 걸려있음 - 바로 매수가능
#     print(i["bid_price"], end="\r") # 매수가 걸려있음 - 바로 매도가능

# upbit = pyupbit.Upbit(key.upbit_access, key.upbit_secret)
# print(upbit.get_balances())
# interval_config = 80
# order_time = (datetime.datetime.now() - datetime.timedelta(minutes=60)).strftime("%Y-%m-%d %H:%M:%S")
#
# if (datetime.datetime.now() - datetime.timedelta(minutes=interval_config)) \
#         < datetime.datetime.strptime(order_time, "%Y-%m-%d %H:%M:%S"):
#     print(datetime.datetime.now())
#     print((datetime.datetime.now() - datetime.timedelta(minutes=interval_config)))
#     print(datetime.datetime.strptime(order_time, "%Y-%m-%d %H:%M:%S"))
#     print("skip")

# import torch
#
#
#
# print(torch.cuda.is_available(),torch.cuda.current_device(),torch.cuda.get_device_name(0))

# def rendering_model_eval(save_name, clf, res, interval, rate, X_train, X_test, y_train, y_test, ticker_data):
#     X_origin, y_origin = train_data_processing(res, interval, rate, render=True)
#
#     test_pred = clf.predict(X_test)
#     train_pred = clf.predict(X_train)
#
#     # View about Train Data - plotly rendering
#     close_train = X_origin[:, -2]
#
#     idx_train_origin = np.argwhere(np.array(y_train) == 1).flatten()
#     idx_train_pred = np.argwhere(np.array(train_pred) == 1).flatten()
#
#     idx_test_origin = np.argwhere(np.array(y_test) == 1).flatten() + len(X_train)
#     idx_test_pred = np.argwhere(np.array(test_pred) == 1).flatten() + len(X_train)
#
#     fig = go.Figure()
#
#     fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_train_origin],
#                              y=ticker_data["close"].iloc[idx_train_origin],
#                              mode="markers", marker=dict(color="black", size=10), opacity=0.7, name="train label"))
#
#     fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_test_origin],
#                              y=ticker_data["close"].iloc[idx_test_origin],
#                              mode="markers", marker=dict(color="black", size=10), opacity=0.7, name="test label"))
#
#     fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_train_pred],
#                              y=ticker_data["close"].iloc[idx_train_pred],
#                              mode="markers", marker=dict(color="red", size=12), opacity=0.7, name="train decision"))
#
#     fig.add_trace(go.Scatter(x=ticker_data["time"].iloc[idx_test_pred],
#                              y=ticker_data["close"].iloc[idx_test_pred],
#                              mode="markers", marker=dict(color="blue", size=12), opacity=0.7, name="test decision"))
#
#     fig.add_trace(go.Scatter(x=ticker_data["time"][:len(close_train)],
#                              y=ticker_data["close"][:len(close_train)],
#                              mode="lines", marker=dict(color="black"), opacity=0.3, name="ticker data"))
#
#     fig.add_trace(go.Scatter(x=ticker_data["time"][:len(close_train)],
#                              y=ticker_data["interval_max"][:len(close_train)],
#                              mode='lines', marker=dict(color="blue"), opacity=0.5, name=f"interval max"))
#
#     fig.update_layout(xaxis_rangeslider_visible=True)
#     fig.show()
#     fig.write_html(save_name)
#
#     print(recall_score(y_test, test_pred))
#     print(recall_score(y_train, train_pred))
#
#     print(precision_score(y_test, test_pred))
#     print(precision_score(y_train, train_pred))
