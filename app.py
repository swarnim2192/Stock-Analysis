import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

from stock_analysis import (
    fetch_prices, make_features, train_model,
    save_model, load_model, latest_prediction
)

st.set_page_config(page_title="Stock Analysis", layout="wide")
st.title("📈 Stock Analysis & Simple ML")
st.caption("Python • Streamlit • yfinance • scikit-learn • Plotly")

with st.sidebar:
    st.header("Controls")
    watchlist = st.multiselect(
        "Select Ticker(s)",
        ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN", "META", "NFLX"],
        default=["AAPL"],
    )
    period = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"], index=2)
    interval = st.selectbox("Interval", ["1d","1h","30m","15m","5m"], index=0)
    retrain = st.button("🔁 Train / Update Model")
    st.caption("Tip: If rate-limited, try again in ~30–60s or change period/interval.")

@st.cache_data(show_spinner=True, ttl=300)
def _cached_prices(ticker: str, p: str, i: str) -> pd.DataFrame:
    return fetch_prices(ticker, period=p, interval=i)

col_left, col_right = st.columns([2, 1], gap="large")

try:
    # --- Watchlist comparison chart (normalized returns) ---
    st.subheader("📊 Watchlist Comparison (returns normalized to 1)")
    compare_data = {}
    for tk in watchlist:
        try:
            df = _cached_prices(tk, period, interval)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            compare_data[tk] = df["Close"]
        except Exception as e:
            st.warning(f"⚠️ {tk}: {e}")
    if compare_data:
        merged = pd.concat(compare_data, axis=1)
        norm = merged / merged.iloc[0]
        fig_compare = px.line(norm, x=norm.index, y=norm.columns, title="Normalized Returns Comparison")
        fig_compare.update_layout(height=420, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_compare, use_container_width=True)

    # --- NEW: Side-by-side price charts for each selected ticker (2 per row) ---
    if compare_data:
        st.subheader("📈 Price charts (per ticker)")
        cols = st.columns(2, gap="large")
        col_idx = 0
        for tk in watchlist:
            df = _cached_prices(tk, period, interval)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            plot_df = df.reset_index()
            time_col = plot_df.columns[0]
            fig_tk = px.line(plot_df, x=time_col, y="Close", title=f"{tk} Close ({period}, {interval})")
            fig_tk.update_layout(height=340, margin=dict(l=0,r=0,t=40,b=0))
            with cols[col_idx]:
                st.plotly_chart(fig_tk, use_container_width=True)
            col_idx = 1 - col_idx  # 0 -> 1 -> 0 ...
            if col_idx == 0 and tk != watchlist[-1]:
                cols = st.columns(2, gap="large")

    # --- Single-ticker analysis (use first selected ticker) ---
    ticker = watchlist[0]
    prices = _cached_prices(ticker, period, interval)
    if isinstance(prices.attrs.get("stale", False), bool) and prices.attrs.get("stale"):
        st.warning("Using cached data due to API rate limit. Try again later or adjust period/interval.")
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    with col_left:
        st.subheader(f"{ticker} Price (focus)")
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

    # --- Evaluation details ---
    with st.expander("📊 Evaluation details (confusion matrix & ROC)"):
        if pipe is not None and not X.empty:
            from sklearn.metrics import confusion_matrix, roc_curve, auc
            n_test = max(1, int(len(X)*0.25))
            X_test, y_test = X.tail(n_test), y.tail(n_test)
            if hasattr(pipe, "predict_proba"):
                y_proba = pipe.predict_proba(X_test)[:, 1]
            else:
                y_scores = pipe.decision_function(X_test)
                y_proba = (y_scores - y_scores.min())/(y_scores.max() - y_scores.min() + 1e-9)
            y_pred = (y_proba >= 0.5).astype(int)

            cm = confusion_matrix(y_test, y_pred, labels=[0,1])
            cm_fig = go.Figure(data=go.Heatmap(
                z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"],
                text=cm, texttemplate="%{text}", colorscale="Blues"
            ))
            cm_fig.update_layout(title="Confusion Matrix", height=320, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(cm_fig, use_container_width=True)

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"))
            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
            roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=320, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(roc_fig, use_container_width=True)
        else:
            st.info("Train the model first to view evaluation plots.")

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
