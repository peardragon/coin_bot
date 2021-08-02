from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json
import requests
import pyupbit
import asyncio
import aiohttp
import pandas as pd
import time
import numpy as np

# now = datetime.strptime(datetime.now(),).strftime()
# prev = (datetime.now() - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
#
# print(now, "\n", prev)

async def get_current_data(ticker, resolution):
    fin_time = int(datetime.now().timestamp() + 9 * 60 * 60)
    start_time = int((datetime.now() - timedelta(minutes=resolution)).timestamp())

    init_ticker = ticker
    string = ticker.replace("-", " ")
    split_strings = string.split()
    reversed_split_strings = list(reversed(split_strings))
    ticker = ' '.join(reversed_split_strings)
    symbol = ticker

    url = f"https://crix-api-tv.upbit.com/v1/crix/tradingview/history?" \
          f"symbol={symbol}&resolution={resolution}&from={int(start_time)}&to={int(fin_time)}"
    # print(url)


    async with aiohttp.ClientSession() as sess:
        async with sess.get(url, headers={'user-agent': 'Mozilla/5.0'}) as res:
            text = await res.text()

    # res = requests.get(url)
    # html = res.text

    soup = BeautifulSoup(text, 'lxml')
    result = soup.find('body').text
    hist_dict = json.loads(result)

    if hist_dict['s'] == 'ok':
        del (hist_dict['s'])
        print(f"{symbol} Min data")

    elif hist_dict['s'] == "no_data":
        print("data 없음. Continue", end='\r')
        return None

    else:
        print(f"Error Occur, not Satisfied URL condition, \n With {start_time}&{fin_time}")
        print(url)

    hist_df = pd.DataFrame.from_dict(hist_dict)

    hist_df['t'] = pd.to_datetime(hist_df['t'], unit='ms')
    hist_df['t'] = pd.DatetimeIndex(hist_df['t']) + timedelta(hours=9)  # 한국시간으로변경

    hist_df = hist_df.rename(
        columns={"t": "time", "o": "open", "l": "low", "h": "high", "c": "close", "v": "volume"})

    df = hist_df
    # na 값에 대한 처리.
    # df['close'] = df.close.fillna(method='ffill')
    # df['open'] = df.open.fillna(df['close'])
    # df['low'] = df.low.fillna(df['close'])
    # df['high'] = df.high.fillna(df['close'])
    # # fill to prior close
    # df['volume'] = df.volume.fillna(0)
    # df = df.fillna(0)

    if df['high'][0] == df['close'][0]:
        print("종가와 최고가 동일")

    print(df)



async def main():
    ticker_list = pyupbit.get_tickers()
    # KRW 추출.
    ticker_list = np.array(ticker_list)[["KRW" in ticker[0:3] for ticker in ticker_list]]
    print(len(ticker_list))
    futures = [asyncio.ensure_future(get_current_data(ticker, resolution=5)) for ticker in ticker_list]
    await asyncio.gather(*futures)

if __name__ == "__main__":

    start = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    end = time.time()
    print(f'time taken: {end-start}')
# 비동기적 처리 3초 vs 동기적 처리 17초.
# KRW 만 craw - 1.96초