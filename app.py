import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.metrics import confusion_matrix, roc_curve, auc

from stock_analysis import (
    fetch_prices, make_features, train_model,
    save_model, load_model, latest_prediction
)

st.set_page_config(page_title="Stock Analysis", layout="wide")

# ----- Header -----
st.title("📈 Stock Analysis & Simple ML")
st.caption("Python • Streamlit • yfinance • scikit-learn • Plotly")

# ----- Dark/Light charts (Plotly only) -----
if "dark_charts" not in st.session_state:
    st.session_state.dark_charts = False
st.session_state.dark_charts = st.toggle("Dark charts", value=st.session_state.dark_charts, help="Toggles Plotly theme (charts only)")

TPL = "plotly_dark" if st.session_state.dark_charts else "plotly"
BG  = "#0e1117" if st.session_state.dark_charts else "white"
FG  = "#e6e6e6" if st.session_state.dark_charts else "#111111"
GRID = "#2a2a2a" if st.session_state.dark_charts else "#e5e5e5"

def style_fig(fig, title=None):
    fig.update_layout(
        template=TPL,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=FG),
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
        title=title or fig.layout.title.text,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

# ----- Top navbar -----
selected = option_menu(
    None,
    ["Home", "Features", "About"],
    icons=["house", "stars", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# ----- One-time welcome banner (on first visit only) -----
if "welcomed" not in st.session_state:
    st.info(
        "👋 **Welcome!** Pick one or more tickers in the sidebar, choose a period & interval, "
        "then click **Train / Update Model** to (re)train. Use **Export CSV** under the table to download data.",
        icon="✅",
    )
    st.session_state.welcomed = True

with st.sidebar:
    st.header("Controls")
    watchlist = st.multiselect(
        "Select Ticker(s)",
        ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN", "META", "NFLX"],
        default=["AAPL"],
        help="Choose one or more stocks to visualize and compare."
    )
    period = st.selectbox(
        "Period",
        ["1mo","3mo","6mo","1y","2y","5y"],
        index=2,
        help="How far back to load data."
    )
    interval = st.selectbox(
        "Interval",
        ["1d","1h","30m","15m","5m"],
        index=0,
        help="Bar size (daily vs intraday). Intraday may hit rate limits sooner."
    )
    retrain = st.button("🔁 Train / Update Model", help="Re-train the model on the most recent data.")
    st.caption("If rate-limited, try again in ~30–60s or change period/interval.")

@st.cache_data(show_spinner=True, ttl=300)
def _cached_prices(ticker: str, p: str, i: str) -> pd.DataFrame:
    return fetch_prices(ticker, period=p, interval=i)

# =================== HOME ===================
if selected == "Home":
    col_left, col_right = st.columns([2, 1], gap="large")
    try:
        # Watchlist normalized comparison
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
            style_fig(fig_compare)
            st.plotly_chart(fig_compare, use_container_width=True)

        # Side-by-side price charts
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
                style_fig(fig_tk)
                with cols[col_idx]:
                    st.plotly_chart(fig_tk, use_container_width=True)
                col_idx = 1 - col_idx
                if col_idx == 0 and tk != watchlist[-1]:
                    cols = st.columns(2, gap="large")

        # Focused single-ticker
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
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Raw Data (tail)")
            st.dataframe(prices.tail(200))

            # ---- Export CSV buttons (Quick Win #3) ----
            csv_bytes = prices.to_csv(index=True).encode()
            st.download_button(
                "📥 Export CSV (price data)",
                data=csv_bytes,
                file_name=f"{ticker}_{period}_{interval}.csv",
                mime="text/csv",
                help="Download the table above as CSV."
            )

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

        # Evaluation expander
        with st.expander("📊 Evaluation details (confusion matrix & ROC)"):
            if pipe is not None and not X.empty:
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
                style_fig(cm_fig, "Confusion Matrix")
                cm_fig.update_layout(height=320)
                st.plotly_chart(cm_fig, use_container_width=True)

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"))
                roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
                style_fig(roc_fig, "ROC Curve")
                roc_fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=320)
                st.plotly_chart(roc_fig, use_container_width=True)
            else:
                st.info("Train the model first to view evaluation plots.")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# =================== FEATURES ===================
elif selected == "Features":
    st.header("✨ Features")
    st.markdown("""
- **Watchlist & compare:** Select multiple tickers and compare normalized returns.
- **Side-by-side charts:** See individual price charts (2 per row) for your selections.
- **Model & metrics:** Logistic Regression with accuracy/precision/recall/F1.
- **Evaluation details:** Confusion Matrix and ROC/AUC.
- **Caching & fallback:** 5-minute cache, Stooq fallback, and last-good parquet snapshot to handle rate limits.
- **Ticker/period/interval controls:** Flexible data views with intraday/daily options.
- **Dark charts toggle:** Switch Plotly theme without changing the whole app theme.
- **Export CSV:** Download the displayed price table as CSV.
- **Guided help:** Tooltips describe each control for new users.
    """)

# =================== ABOUT ===================
else:
    st.header("ℹ️ About")
    st.markdown("""
This portfolio app uses **Streamlit** (UI), **yfinance/Stooq** (data), **pandas** (ETL), **scikit-learn** (ML), and **Plotly** (charts).
It predicts next-bar direction with engineered features (returns, MA ratios, RSI, Bollinger Z).
""")
