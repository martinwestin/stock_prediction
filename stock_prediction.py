import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
import data


def new_model(train_x: np.ndarray, train_y: np.ndarray):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(50))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    model.fit(train_x, train_y, epochs=25)

    model.save(os.path.join(sys.path[0], "stock_model.h5"))


def predict_new_stock(ticker):
    if "stock_model.h5" in os.listdir(sys.path[0]):
        model = tf.keras.models.load_model("stock_model.h5")
        price_data = data.yf.download(ticker, interval="1mo")
        test_data = data.SCALER.fit_transform(price_data["Close"].values.reshape(-1, 1))
        test_data = np.array(list(filter(lambda x: str(x[0]) != "nan", test_data)))
        test_data = test_data[len(test_data) - 12:]
        test_data = test_data.reshape(1, 12, 1)
        predicted = model.predict(test_data)
        predicted = data.SCALER.inverse_transform(predicted)
        return predicted
    else:
        x, y = data.download(data.STOCKS)
        new_model(x, y)
        predict_new_stock(ticker)


def ask_retrain():
    try:
        tf.keras.models.load_model("stock_model.h5")
        retrain = input("Model found. Would you like to retrain the model anyway (Y/N)? ").lower() == "y"
        if retrain:
            x, y = data.download(data.STOCKS)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
            new_model(x_train, y_train)

    except IOError:
        print("No model found. Training...")
        x, y = data.download(data.STOCKS)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        new_model(x_train, y_train)


if __name__ == '__main__':
    ask_retrain()
    x, y = data.download(data.STOCKS)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    model = tf.keras.models.load_model("stock_model.h5")
    predict = model.predict(x_test)
    predict = data.SCALER.inverse_transform(predict)
    y_test = data.SCALER.inverse_transform(y_test)
    print("Running tests...")
    for i in range(len(predict)):
        print(f"Predict: {predict[i]}, Actual: {y_test[i]}")
