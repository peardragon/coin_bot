import websockets
import asyncio
import json


async def upbit_ws_client(ticker):
    uri = "wss://api.upbit.com/websocket/v1"

    async with websockets.connect(uri) as websocket:
        subscribe_fmt = [
            {"ticket":"test"},
            {
                "type": "ticker",
                "codes": [ticker]
            },
            # {"format":"SIMPLE"}
        ]
        # ["KRW-BTC"] : ticker example
        subscribe_data = json.dumps(subscribe_fmt)
        await websocket.send(subscribe_data)

        # while True:
        data = await websocket.recv()
        recv_data = json.loads(data)

    # print(json.loads(data))

    return recv_data


async def main(ticker):
    data = await upbit_ws_client(ticker)
    return data


def run_wm(input_list):
    ticker = input_list[0]
    options = input_list[1]
    data = asyncio.run(main(ticker))

    return [data, options]


# asyncio.run(main(["KRW-BTC"]))

from multiprocessing import Process, Queue
import multiprocessing
import time
if __name__ == '__main__':

    start = time.time()

    ticker_1 = 'KRW-BTC'
    ticker_2 = 'BTC-ETH'
    ticker_3 = 'KRW-ETH'
    pool = multiprocessing.Pool(processes=8)
    input_list = [[ticker_1, "ask"], [ticker_2, "ask"], [ticker_3, "ask"]]
    returns = pool.map(run_wm, input_list)
    pool.close()

    print("returns:", returns)

    print("time :", time.time() - start)

# if __name__ == '__main__':
#
#     start = time.time()
#
#     ticker_1 = 'KRW-BTC'
#     ticker_2 = 'BTC-ETH'
#     ticker_3 = 'KRW-ETH'
#
#     temp = []
#     input_list = [[ticker_1, "ask"], [ticker_2, "ask"], [ticker_3, "ask"]]
#     for i in input_list:
#         temp.append(run_wm(i))
#
#     returns = temp
#
#     print("returns:", returns)
#
#     print("time :", time.time() - start)