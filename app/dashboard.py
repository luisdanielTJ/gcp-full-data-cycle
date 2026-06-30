from datetime import datetime, timezone

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adapters.config import API_BASE_URL

st.set_page_config(page_title="crypto-edge", layout="centered")

ASSETS = ["XBTUSD", "ETHUSD"]
ASSET_LABELS = {"XBTUSD": "BTC", "ETHUSD": "ETH"}

client = httpx.Client(base_url=API_BASE_URL, timeout=10.0)


def _get(path: str):
    response = client.get(path)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def _ago(timestamp_str: str) -> str:
    ts = pd.Timestamp(timestamp_str)
    delta = datetime.now(timezone.utc) - ts.to_pydatetime()
    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m ago"


st.title("crypto-edge")

st.header("Signals")
cols = st.columns(2)
for col, asset in zip(cols, ASSETS):
    with col:
        st.subheader(ASSET_LABELS[asset])
        signal = _get(f"/signals/{asset}")
        if signal is None:
            st.info("No signal yet")
            continue
        badge = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}[signal["signal"]]
        st.metric(f"{badge} {signal['signal']}", f"{signal['confidence']:.0%} confidence")
        st.caption(_ago(signal["predicted_at"]))

        narration = _get(f"/narration/{asset}")
        if narration:
            st.write(narration["narration"])

        st.link_button(
            "Trade on Binance",
            f"https://www.binance.com/en/trade/{ASSET_LABELS[asset]}_USDT",
        )

st.header("Charts")
for asset in ASSETS:
    st.subheader(ASSET_LABELS[asset])
    candles = _get(f"/ohlcv/{asset}")
    if not candles:
        st.info("No chart data yet")
        continue
    df = pd.DataFrame(candles)
    df["open_time"] = pd.to_datetime(df["open_time"])
    fig = go.Figure(data=[go.Candlestick(
        x=df["open_time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    )])

    history = _get(f"/signals/{asset}/history") or []
    if history:
        hist_df = pd.DataFrame(history)
        hist_df["predicted_at"] = pd.to_datetime(hist_df["predicted_at"])
        cutoff = df["open_time"].min()
        hist_df = hist_df[hist_df["predicted_at"] >= cutoff]
        merged = pd.merge_asof(
            hist_df.sort_values("predicted_at"), df.sort_values("open_time"),
            left_on="predicted_at", right_on="open_time", direction="nearest",
        )
        for signal_type, color in (("BUY", "green"), ("SELL", "red")):
            points = merged[merged["signal"] == signal_type]
            if not points.empty:
                symbol = "triangle-up" if signal_type == "BUY" else "triangle-down"
                fig.add_trace(go.Scatter(
                    x=points["open_time"], y=points["close"], mode="markers",
                    marker=dict(color=color, size=10, symbol=symbol),
                    name=signal_type,
                ))

    st.plotly_chart(fig, use_container_width=True)

st.header("Feature Breakdown")
for asset in ASSETS:
    signal = _get(f"/signals/{asset}")
    if signal is None:
        continue
    import json
    top5 = json.loads(signal["shap_top5"])
    if not top5:
        continue
    st.subheader(ASSET_LABELS[asset])
    bar_df = pd.DataFrame(top5)
    colors = ["green" if v > 0 else "red" for v in bar_df["value"]]
    fig = go.Figure(go.Bar(
        x=bar_df["value"], y=bar_df["feature"], orientation="h", marker_color=colors,
    ))
    st.plotly_chart(fig, use_container_width=True)

st.header("Sentiment Feed")
sentiment = _get("/sentiment")
if sentiment:
    st.dataframe(pd.DataFrame(sentiment)[["source", "sentiment", "title"]])
else:
    st.info("No sentiment data yet")

st.header("Trade Journal")

with st.form("log_trade"):
    asset = st.selectbox("Asset", ASSETS, format_func=lambda a: ASSET_LABELS[a])
    direction = st.selectbox("Direction", ["LONG", "SHORT"])
    entry_price = st.number_input("Entry price", min_value=0.0)
    amount_usd = st.number_input("Amount (USD)", min_value=0.0)
    default_opened_at = datetime.now(timezone.utc).isoformat()
    opened_at = st.text_input("Opened at (ISO timestamp)", value=default_opened_at)
    submitted = st.form_submit_button("Log trade")
    if submitted:
        client.post("/trades", json={
            "asset": asset, "direction": direction, "entry_price": entry_price,
            "amount_usd": amount_usd, "opened_at": opened_at,
        })
        st.rerun()

st.subheader("Open positions")
positions = _get("/positions")
if positions:
    pos_df = pd.DataFrame(positions)
    st.dataframe(pos_df[["asset", "direction", "entry_price", "current_price", "unrealized_pnl"]])
    for pos in positions:
        with st.expander(f"Close {ASSET_LABELS.get(pos['asset'], pos['asset'])} position"):
            exit_price = st.number_input("Exit price", min_value=0.0, key=f"exit_{pos['id']}")
            if st.button("Close", key=f"close_{pos['id']}"):
                client.patch(f"/trades/{pos['id']}/close", json={"exit_price": exit_price})
                st.rerun()
else:
    st.info("No open positions")

st.subheader("Closed trades")
all_trades = _get("/trades") or []
closed = [t for t in all_trades if t.get("closed_at")]
if closed:
    st.dataframe(pd.DataFrame(closed)[["asset", "direction", "entry_price", "exit_price"]])
else:
    st.info("No closed trades yet")

st.subheader("Performance summary")
performance = _get("/performance")
if performance:
    perf_cols = st.columns(5)
    perf_cols[0].metric("Total P&L", f"${performance['total_pnl']:.2f}")
    win_rate = performance["win_rate"]
    perf_cols[1].metric("Win rate", f"{win_rate:.0%}" if win_rate is not None else "N/A")
    best = performance["best_trade_pnl"]
    worst = performance["worst_trade_pnl"]
    perf_cols[2].metric("Best trade", f"${best:.2f}" if best is not None else "N/A")
    perf_cols[3].metric("Worst trade", f"${worst:.2f}" if worst is not None else "N/A")
    sig_acc = performance["signal_accuracy"]
    perf_cols[4].metric("Signal accuracy", f"{sig_acc:.0%}" if sig_acc is not None else "N/A")
