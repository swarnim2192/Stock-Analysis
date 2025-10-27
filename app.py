import streamlit as st
import pandas as pd
import plotly.express as px
from stock_analysis import (
    fetch_prices, make_features, train_model,
    save_model, load_model, latest_prediction
)

st.set_page_config(page_title="Stock Analysis", layout="wide")
st.title("📈 Stock Analysis & Simple ML")
st.caption("Python • Streamlit • yfinance • scikit-learn • Plotly")

with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    period = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"], index=2)
    interval = st.selectbox("Interval", ["1d","1h","30m","15m","5m"], index=0)
    retrain = st.button("🔁 Train / Update Model")

@st.cache_data(show_spinner=True)
def _cached_prices(t: str, p: str, i: str) -> pd.DataFrame:
    return fetch_prices(t, period=p, interval=i)

col_left, col_right = st.columns([2, 1], gap="large")

try:
    prices = _cached_prices(ticker, period, interval)

    with col_left:
        st.subheader(f"{ticker} Price")
        df_plot = prices.reset_index()
        time_col = df_plot.columns[0]
        fig = px.line(df_plot, x=time_col, y="Close", title=f"{ticker} Close ({period}, {interval})")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Raw Data (tail)")
        st.dataframe(prices.tail(200))

    with col_right:
        st.subheader("Model")
        X, y, feats = make_features(prices)
        pipe = load_model("model.joblib")

        if retrain or pipe is None:
            with st.spinner("Training model..."):
                result = train_model(X, y)
                pipe = result.pipeline
                save_model(result, "model.joblib")
                st.success("Model trained and saved.")
                st.metric("Accuracy", f"{result.metrics['accuracy']:.3f}")
                st.metric("Precision", f"{result.metrics['precision']:.3f}")
                st.metric("Recall", f"{result.metrics['recall']:.3f}")
                st.metric("F1", f"{result.metrics['f1']:.3f}")
                st.caption(f"Train n={result.metrics['n_train']}, Test n={result.metrics['n_test']}")
        else:
            st.success("Loaded saved model. Click **Train / Update Model** to retrain.")

        pred = latest_prediction(pipe, X)
        if pred is not None:
            st.metric("Next-bar Direction (model)", "⬆️ Up" if pred == 1 else "⬇️ Down")
        else:
            st.warning("Not enough data for a prediction yet.")

        st.divider()
        st.write("**Features used:**", ", ".join(feats))
        st.caption("Label = 1 if next return > 0 else 0.")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
