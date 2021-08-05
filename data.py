import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


STOCKS = [
    "MSFT", "XOM", "GOOG", "JNJ", "AMZN",
    "PNC", "TSLA", "NVDA", "PFE", "J",
    "GME", "GILD", "MMM", "ACN", "APD",
    "AAPL", "DIS", "CF", "RE", "IBM",
    "ABC", "CE", "DE", "F", "JCI"
]
SCALER = MinMaxScaler(feature_range=(0, 1))


def download(tickers, period: str="10y", price_type: str="Close"):
    x, y, = [], []
    for stock in tickers:
        data = yf.download(stock, period=period, interval="1mo")[price_type].values.reshape(-1, 1)
        data = SCALER.fit_transform(data.reshape(-1, 1))
        data = np.array(list(filter(lambda x: str(x[0]) != "nan", data)))
        for i in range(12, len(data)):
            x.append(data[i - 12: i])
            y.append(data[i])

    x, y = np.array(x), np.array(y)
    return x, y
