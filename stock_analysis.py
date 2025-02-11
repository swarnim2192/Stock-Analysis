import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Test to ensure everything works
print("Libraries imported successfully!")

# Step 1: Fetch Historical Stock Data
data = yf.download('AAPL', start='2015-01-01', end='2025-01-01')

# Display the first 5 rows to understand the structure
print(data.head())

# Step 2: Save the Data Locally (Optional)
data.to_csv('AAPL_stock_data.csv')

# Load the saved data (debugging added)
# Checking CSV column headers to debug the 'Date' issue
temp_data = pd.read_csv('AAPL_stock_data.csv')
print("CSV Columns:", temp_data.columns)

# Handling potential missing 'Date' column issue
if 'Date' in temp_data.columns:
    data = pd.read_csv('AAPL_stock_data.csv', index_col='Date', parse_dates=True)
else:
    data = pd.read_csv('AAPL_stock_data.csv')
    data['Date'] = pd.to_datetime(data.index)
    data.set_index('Date', inplace=True)

# Step 3: Data Cleaning & Preprocessing
# Check for Missing Values
print(data.isnull().sum())

# Handle Missing Data
# Dropping missing values
data = data.dropna()

# Verify Data Types
print(data.dtypes)

# Add Moving Averages for Trend Analysis
data['MA50'] = data['Close'].rolling(window=50).mean()   # 50-day moving average
data['MA200'] = data['Close'].rolling(window=200).mean() # 200-day moving average

# Step 4: Exploratory Data Analysis (EDA)
# Plotting the Closing Price with Moving Averages
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
# Preparing data for Linear Regression
# Converting Date to an ordinal format for regression analysis
data['Date_ordinal'] = pd.to_datetime(data.index).map(pd.Timestamp.toordinal)

# Defining features (X) and target (y)
X = data[['Date_ordinal']]  # Feature: Date in ordinal format
y = data['Close']           # Target: Closing price

# Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Making predictions
predictions = model.predict(X)

# Evaluating the model
mse = mean_squared_error(y, predictions)
print(f'Mean Squared Error (MSE): {mse}')

# Plotting Actual vs Predicted Prices
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
data['SMA30'] = data['Close'].rolling(window=30).mean()  # 30-day simple moving average

# Plotting the Moving Average Forecast
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Actual Price', color='blue')
plt.plot(data['SMA30'], label='30-Day Moving Average', color='orange')
plt.legend()
plt.title('Stock Price with 30-Day Moving Average Forecast')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid()
plt.show()

# Predicting Future Stock Prices for the Next 30 Days
future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

# Predict future prices
future_predictions = model.predict(future_dates_ordinal)

# Plotting the Predictions
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Actual Price', color='blue')
plt.plot(future_dates, future_predictions, label='Predicted Future Price', linestyle='dashed', color='red')
plt.legend()
plt.title('Stock Price Prediction for the Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid()
plt.show()

from sklearn.metrics import mean_squared_error
import math

# Calculating RMSE
rmse = math.sqrt(mean_squared_error(y, predictions))
print(f'Root Mean Squared Error (RMSE): {rmse}')
