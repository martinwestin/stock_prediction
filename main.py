import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
import stock_prediction
import yfinance as yf


class PlaceholderEntry(tk.Entry):
    def __init__(self, master, placeholder_value: str):
        super().__init__(master)
        self.placeholder = placeholder_value
        self.insert(0, self.placeholder)
        self.configure(state=tk.DISABLED)
        self.bind("<Button-1>", self.click)
        self.bind("<Meta_L><BackSpace>", self.delete_all)
        master.bind("<Button-1>", self.click_master)

    def click(self, event):
        if self.get() == self.placeholder:
            self.configure(state=tk.NORMAL)
            self.delete(0, tk.END)

    def click_master(self, event):
        if len(self.get()) == 0:
            self.insert(0, self.placeholder)
            self.configure(state=tk.DISABLED)

    def delete_all(self, event):
        self.delete(0, tk.END)


class App(tk.Tk):
    WIDTH, HEIGHT = (600, 500)

    def __init__(self):
        super().__init__()
        self.title("Stock Prediction by Martin Westin")
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.main_frame = tk.Frame(self, width=self.WIDTH*0.9, height=self.HEIGHT*0.6, bg="lightblue")
        self.place_widgets()
        self.ticker = ""

    def place_widgets(self):
        self.main_frame.place(relx=0.05, rely=0.05)
        self.new_ticker = PlaceholderEntry(self.main_frame, "Enter new stock...")
        self.new_ticker.place(relx=0.05, rely=0.05)
        self.new_ticker.bind("<Return>", self.predict_new_ticker)

        self.prices = tk.Listbox(self.main_frame, bg="lightgreen")
        self.prices.place(relx=0.05, rely=0.2, relheight=0.7, relwidth=0.8)
        scrollbar = tk.Scrollbar(self.prices)
        scrollbar.place(relx=0.95, rely=0.05, relheight=0.9)
        self.prices.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.prices.yview)

    def predict_new_ticker(self, event):
        ticker = str(self.new_ticker.get()).upper()
        try:
            prediction = stock_prediction.predict_new_stock(ticker)
            data = yf.download(ticker, period="1y")
            prices = data["Close"].values
            history = data.index

            prices = np.array(list(filter(lambda x: str(x) != "nan", prices)))
            history = np.array(list(filter(lambda x: str(list(prices)[list(history).index(x)]) != "nan", history)))
            insert_prices = list(map(lambda x: (x, history[list(prices).index(x)]), prices))
            self.insert_values(insert_prices, prediction)

            plt.plot(history, prices)
            plt.title(f"Share price of stock with ticker {ticker}")
            plt.xlabel("Time")
            plt.ylabel("Price in USD")
            plt.show()

        except ValueError:
            messagebox.showerror("Stock Error", f"Stock with ticker {ticker} could not be found.")

    def insert_values(self, values, prediction):
        self.prices.delete(0, tk.END)
        for price in reversed(values):
            insert_data = f"{str(price[1]).split(' ')[0]}: {price[0]}"
            self.prices.insert(list(reversed(values)).index(price), insert_data)

        self.prices.insert(0, f"Prediction for next month: ${prediction[0][0]}")


if __name__ == '__main__':
    app = App()
    app.mainloop()
