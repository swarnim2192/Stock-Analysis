import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# App Title
st.title('📈 Stock Market Analysis Dashboard')

# User Input for Ticker Symbol
ticker = st.text_input('Enter Stock Ticker:', 'AAPL')

# Fetch Stock Data
data = yf.download(ticker, start='2015-01-01', end='2025-01-01')

# Display Raw Data
st.subheader('📊 Raw Stock Data')
st.write(data.tail())

# Plotting Closing Price
st.subheader('📈 Closing Price Over Time')
st.line_chart(data['Close'])

# Adding Moving Averages
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# Plotting Moving Averages
st.subheader('Moving Averages (50 & 200 Days)')
st.line_chart(data[['Close', 'MA50', 'MA200']])

# Volume Analysis
st.subheader('Trading Volume Over Time')
st.bar_chart(data['Volume'])
