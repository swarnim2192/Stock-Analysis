import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Test to ensure everything works
print("Libraries imported successfully!")
# Plotting the Closing Price with Moving Averages
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.plot(data['MA50'], label='50-Day Moving Average', color='red')
plt.plot(data['MA200'], label='200-Day Moving Average', color='green')
plt.legend()
plt.title('Apple Stock Price Trend with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid()
plt.show()
