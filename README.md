# Stock prediction program in python
A Recurrent Neural Network with LSTM using Tensorflow in python
that is predicts what the price of a specific stock will be in one month.

## About the model
This is a Recurrent Neural Network using multiple LSTM layers.
The model is trained on historical price data for different stocks in the S&P500.
This means that it often times predicts small price changes, as companies
that are listed in the S&P500 are big companies whose stock prices don't
tend to fluctuate very much. On testing data, the model has managed to predict
pretty accurately, and has in many cases predicted just a few cents or dollars
below or above the actual price.

## Files
main.py is the main file of this project, and you can run this to interact
with the model through a GUI I made with tkinter. Here, you can look up
historical data on any stock, and also get the predicted price in one month.
data.py is used to collect data on different stocks and format the prices.
stock_prediction.py is the file where the actual model is created and where
predictions are made. stock_model.h5 contains a pretrained model,
do not edit this file manually.

## Usage
Do not use this program as financial/investment advice, as this is not
a professional stock prediction program.
