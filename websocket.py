import websockets
import asyncio
import json
import uuid
import multiprocessing as mp


def get_arbitrage_list():
    import pyupbit
    ticker_list = pyupbit.get_tickers()
    arbitrage_list = []
    for i in ticker_list:
        a = i.split("-")
        if a[0] == 'KRW':
            #         print("i",i)
            for j in ticker_list:
                b = j.split("-")
                if a[1] == b[0] and b[0] != "KRW":
                    #                 print("j:",j)
                    for k in ticker_list:
                        c = k.split("-")
                        if b[1] == c[1] and c[0] == "KRW":
                            #                         print("k:", k)
                            arbitrage_list.append([i, j, k])

    return arbitrage_list

# Ref : https://github.com/sharebook-kr/pyupbit/blob/master/pyupbit/websocket_api.py

class WebSocketManager(mp.Process):
    """웹소켓을 관리하는 클래스
        사용 예제:
            >> wm = WebSocketManager("ticker", ["BTC_KRW"])
            >> for i in range(3):
                data = wm.get()
                print(data)
            >> wm.terminate()
        주의 :
           재귀적인 호출을 위해 다음의 guard를 반드시 추가해야 한다.
           >> if __name__ == "__main__"
    """
    def __init__(self, type: str, codes: list, qsize: int=1000):
        """웹소켓을 컨트롤하는 클래스의 생성자
        Args:
            type   (str           ): 구독 메시지 종류 (ticker/trade/orderbook)
            codes  (list          ): 구독할 암호 화폐의 리스트 [BTC_KRW, ETH_KRW, …]
            qsize  (int , optional): 메시지를 저장할 Queue의 크기
        """
        self.__q = mp.Queue(qsize)
        self.alive = False

        self.type = type
        self.codes = codes

        super().__init__()

    async def __connect_socket(self):
        uri = "wss://api.upbit.com/websocket/v1"
        async with websockets.connect(uri, ping_interval=None) as websocket:
            data = [{"ticket": str(uuid.uuid4())[:6]}, {"type": self.type, "codes": self.codes}]
            await websocket.send(json.dumps(data))

            while self.alive:
                recv_data = await websocket.recv()
                recv_data = recv_data.decode('utf8')
                self.__q.put(json.loads(recv_data))

    def run(self):
        self.__aloop = asyncio.get_event_loop()
        self.__aloop.run_until_complete(self.__connect_socket())

    def get(self):
        if self.alive == False:
            self.alive = True
            self.start()
        return self.__q.get()

    def terminate(self):
        self.alive = False
        super().terminate()


def ticker_orderbook(ticker, option='ask'):
    if option == "ask":
        option = "ask_price"
        size = "ask_size"
    else:
        option = "bid_price"
        size = "bid_size"

    recv_data = WebSocketManager("orderbook", [f"{ticker}.1"])
    data = recv_data.get()["orderbook_units"][-1][option]
    size = recv_data.get()["orderbook_units"][-1][size]

    return data, size


if __name__ == "__main__":
    arbitrage_list = get_arbitrage_list()

    from multiprocessing import Process, Queue
    import multiprocessing

    for i in arbitrage_list:
        ticker_1 = i[0]
        ticker_2 = i[1]
        ticker_3 = i[2]

        pool = multiprocessing.Pool(processes=3)
        returns = pool.map(ticker_orderbook, zip([ticker_1, ticker_2, ticker_3], ["ask", "bid", "bid"]))
        print("returns:", returns)

        # Q = Queue
        # p1 = Process(target=ticker_orderbook(ticker_1, option="ask"))
        # p2 = Process(target=ticker_orderbook(ticker_2, option="bid"))
        # p3 = Process(target=ticker_orderbook(ticker_3, option="bid"))
        #
        # p1.start()
        # p2.start()
        # p3.start()
        #
        # p1.join()
        # p2.join()
        # p3.join()

        # wm = WebSocketManager("orderbook", i)
        # print(i)
        # for j in range(4):
        #     data = wm.get()
        #     print(data, type(data))
        # # for i in range(4):
        # #     data = wm.get()
        # #     orderbook_list = data["orderbook_units"]
        # #     print(data)