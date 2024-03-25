import pandas as pd
import yfinance as yf

tickers = [
    "ADANIPORTS.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BPCL.NS",
    "BHARTIARTL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DIVISLAB.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "HDFC.NS",
    "ICICIBANK.NS",
    "ITC.NS",
    "IOC.NS",
    "INFY.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NTPC.NS",
    "NESTLEIND.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SHREECEM.NS",
    "SBIN.NS",
    "SUNPHARMA.NS",
    "TCS.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TECHM.NS",
    "TITAN.NS",
    "UPL.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
]


def load_data(stock: str):
    data = yf.Ticker(stock)
    df = data.history(period="max")
    df["PrevHigh"] = df["High"]
    df["PrevLow"] = df["Low"]
    df["PrevClose"] = df["Close"]
    df["PrevVolume"] = df["Volume"]
    df["PrevHigh"] = df["PrevHigh"].shift(periods=1)
    df["PrevLow"] = df["PrevLow"].shift(periods=1)
    df["PrevClose"] = df["PrevClose"].shift(periods=1)
    df["PrevVolume"] = df["PrevVolume"].shift(periods=1)
    df = df.drop(['Dividends', 'Stock Splits'], axis=1)
    df = df[1:]

    df.to_csv(f"ticker_data/{stock}.csv", index=False)


for ticker in tickers:
    load_data(ticker)
