import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def get_prices(share_symbol, start_date, end_date, cache_filename="stock_prices.npy"):
    try:
        stock_prices = np.load(cache_filename)
    except IOError:
        stock_data = yf.download(share_symbol, start=start_date, end=end_date)
        stock_prices = stock_data['Open'].values
        np.save(cache_filename, stock_prices)
    return stock_prices.astype(float)

def plot_prices(prices):
    plt.figure(figsize=(8, 4))
    plt.title('Opening Stock Prices')
    plt.xlabel('Day')
    plt.ylabel('Price ($)')
    plt.plot(prices)
    plt.savefig('prices.png')
