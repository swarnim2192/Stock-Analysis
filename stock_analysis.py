import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# Function to fetch stock data
def get_stock_data(ticker):
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01')
    data['MA50'] = data['Close'].rolling(window=50).mean()   # 50-day moving average
    data['MA200'] = data['Close'].rolling(window=200).mean() # 200-day moving average
    return data

# Run this part only if the script is executed directly
if __name__ == "__main__":
    # Test to ensure everything works
    print("Libraries imported successfully!")

    # Fetch data for AAPL
    data = get_stock_data('AAPL')
    print(data.head())

    # Step 3: Data Cleaning & Preprocessing
    # Check for Missing Values
    print(data.isnull().sum())

    # Handle Missing Data
    data = data.dropna()

    # Verify Data Types
    print(data.dtypes)

    # Step 4: Exploratory Data Analysis (EDA)
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['MA50'], label='50-Day Moving Average', color='red')
    plt.plot(data['MA200'], label='200-Day Moving Average', color='green')
    plt.legend()
    plt.title('Apple Stock Price Trend (With Moving Averages)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid()
    plt.show()

    # Volume vs Close Price Analysis
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data['Volume'], y=data['Close'])
    plt.title('Volume vs Closing Price Relationship')
    plt.xlabel('Volume')
    plt.ylabel('Close Price')
    plt.grid()
    plt.show()

    # Correlation Matrix
    correlation = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='viridis')
    plt.title('Correlation Matrix')
    plt.show()

    # Step 5: Predictive Modeling - Linear Regression
    data['Date_ordinal'] = pd.to_datetime(data.index).map(pd.Timestamp.toordinal)
    X = data[['Date_ordinal']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error (MSE): {mse}')

    rmse = math.sqrt(mean_squared_error(y, predictions))
    print(f'Root Mean Squared Error (RMSE): {rmse}')

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, y, label='Actual Price', color='blue')
    plt.plot(data.index, predictions, label='Predicted Price', linestyle='dashed', color='red')
    plt.legend()
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid()
    plt.show()

    # Moving Average Forecasting
    data['SMA30'] = data['Close'].rolling(window=30).mean()
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Actual Price', color='blue')
    plt.plot(data['SMA30'], label='30-Day Moving Average', color='orange')
    plt.legend()
    plt.title('Stock Price with 30-Day Moving Average Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid()
    plt.show()

    # Predicting Future Stock Prices
    future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
    future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    future_predictions = model.predict(future_dates_ordinal)

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Actual Price', color='blue')
    plt.plot(future_dates, future_predictions, label='Predicted Future Price', linestyle='dashed', color='red')
    plt.legend()
    plt.title('Stock Price Prediction for the Next 30 Days')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid()
    plt.show()
