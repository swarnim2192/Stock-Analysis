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
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    period = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"], index=2)
    interval = st.selectbox("Interval", ["1d","1h","30m","15m","5m"], index=0)
    retrain = st.button("🔁 Train / Update Model")
    st.caption("Tip: If rate-limited, try again in ~30–60s or change period/interval.")

@st.cache_data(show_spinner=True, ttl=300)
def _cached_prices(t: str, p: str, i: str) -> pd.DataFrame:
    return fetch_prices(t, period=p, interval=i)

col_left, col_right = st.columns([2, 1], gap="large")

try:
    prices = _cached_prices(ticker, period, interval)

    # Warn if using disk fallback
    if isinstance(prices.attrs.get("stale", False), bool) and prices.attrs.get("stale"):
        st.warning("Using last-good cached data due to API rate limit. Try again shortly or adjust period/interval.")

    # Ensure flat columns
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    # LEFT: Price chart + table
    with col_left:
        st.subheader(f"{ticker} Price")
        df_plot = prices.reset_index()
        time_col = df_plot.columns[0]
        fig = px.line(df_plot, x=time_col, y="Close", title=f"{ticker} Close ({period}, {interval})")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Raw Data (tail)")
        st.dataframe(prices.tail(200))

    # RIGHT: Model + metrics
    with col_right:
        st.subheader("Model")
        X, y, feats = make_features(prices)
        pipe = load_model("model.joblib")

        if retrain or pipe is None:
            with st.spinner("Training model..."):
                result = train_model(X, y)  # shuffle=False inside
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

    # ---- Evaluation Details (Confusion Matrix + ROC) ----
    with st.expander("📊 Evaluation details (confusion matrix & ROC)"):
        if pipe is not None and not X.empty:
            # Use the same split size as in train_model (test_size=0.25, no shuffle)
            n_test = max(1, int(len(X) * 0.25))
            X_test = X.tail(n_test)
            y_test = y.tail(n_test)

            # Predictions
            if hasattr(pipe, "predict_proba"):
                y_proba = pipe.predict_proba(X_test)[:, 1]
            else:
                # Fallback: decision_function scaled to [0,1] if not available
                y_scores = pipe.decision_function(X_test)
                y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-9)
            y_pred = (y_proba >= 0.5).astype(int)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred, labels=[0,1])
            cm_fig = go.Figure(data=go.Heatmap(
                z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"],
                text=cm, texttemplate="%{text}", hoverinfo="skip", colorscale="Blues"
            ))
            cm_fig.update_layout(title="Confusion Matrix", height=320, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(cm_fig, use_container_width=True)

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"))
            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
            roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                                  height=320, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(roc_fig, use_container_width=True)
        else:
            st.info("Train the model first to view evaluation plots.")

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
