import streamlit as st
from stock_analysis import get_stock_data
import pandas as pd
import matplotlib.pyplot as plt

# App Title
st.title('📈 Stock Market Analysis Dashboard')

# User Input for Ticker Symbol
ticker = st.text_input('Enter Stock Ticker:', 'AAPL')

# Fetch Stock Data using the imported function
data = get_stock_data(ticker)

# Display Raw Data
st.subheader('📊 Raw Stock Data')
st.write(data.tail())

# Ensure Moving Averages are Calculated
if 'MA50' not in data.columns:
    data['MA50'] = data['Close'].rolling(window=50).mean()

if 'MA200' not in data.columns:
    data['MA200'] = data['Close'].rolling(window=200).mean()

# Flatten MultiIndex if Present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() for col in data.columns]

# Debugging - Check Available Columns
st.write("Columns Available:", data.columns)

# Plotting Closing Price
st.subheader('📈 Closing Price Over Time')
st.line_chart(data['Close'])

# Plotting Moving Averages
st.subheader('Moving Averages (50 & 200 Days)')
st.line_chart(data[['Close', 'MA50', 'MA200']])

# Volume Analysis
st.subheader('Trading Volume Over Time')
st.bar_chart(data['Volume'])
