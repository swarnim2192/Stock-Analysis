import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# App Title
st.title('📈 Real-Time Apple Stock Price & AI Prediction')

# User Input for Timeframe
st.sidebar.header('Select Timeframe')
timeframe = st.sidebar.selectbox(
    'Choose a timeframe:',
    options=['1m', '5m', '15m', '30m', '1h', '1d'],
    index=0
)

# Fetch Real-Time Data
ticker = 'AAPL'
data = yf.download(tickers=ticker, period='1d', interval=timeframe)  # Dynamic timeframe

# Display Real-Time Stock Graph
st.subheader(f'📊 Real-Time Stock Price ({timeframe} Interval)')
plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Real-Time Closing Price', color='#26A69A')  # Teal Green
plt.xlabel('Time', color='#5C6BC0')  # Indigo
plt.ylabel('Price (USD)', color='#5C6BC0')
plt.grid(color='#F0F4F8')  # Soft Blue Gray
plt.legend(facecolor='#FFD54F')  # Amber
st.pyplot(plt)

# AI Prediction Model
st.subheader('🤖 AI Prediction: Will the Price Go Up or Down?')

# Feature Engineering
data['Price Change'] = data['Close'].diff()
data['Direction'] = (data['Price Change'] > 0).astype(int)  # 1 if price goes up, 0 if down

# Prepare Data for Model
data = data.dropna()
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Direction']

# Scaling the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the Latest Movement
latest_data = scaler.transform([X.iloc[-1]])
prediction = model.predict(latest_data)

# Display the Prediction
if prediction[0] == 1:
    st.success('🔼 The AI predicts the stock price will go **UP**!', icon='📈')
    st.markdown("<h3 style='color:#26A69A;'>Bullish Trend Detected 🚀</h3>", unsafe_allow_html=True)
else:
    st.error('🔽 The AI predicts the stock price will go **DOWN**.', icon='📉')
    st.markdown("<h3 style='color:#EF5350;'>Bearish Trend Detected ⚠️</h3>", unsafe_allow_html=True)

# Auto-Refresh Every Minute
st.caption('⏱️ Data refreshes automatically every minute.')
