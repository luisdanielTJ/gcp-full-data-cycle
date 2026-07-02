import json
import uuid
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adapters import get_warehouse
from app.pnl import match_signal_for_trade, summarize_performance
from app.positions import enrich_open_positions

st.set_page_config(page_title="crypto-edge", layout="centered")

ASSETS = ["XBTUSD", "ETHUSD"]
ASSET_LABELS = {"XBTUSD": "BTC", "ETHUSD": "ETH"}


@st.cache_resource
def _warehouse():
    return get_warehouse()


warehouse = _warehouse()


def _ago(ts) -> str:
    delta = datetime.now(timezone.utc) - pd.Timestamp(ts).to_pydatetime()
    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m ago"


def _latest_for_asset(df: pd.DataFrame, asset: str, sort_col: str) -> pd.Series | None:
    if df.empty or "asset" not in df.columns:
        return None
    rows = df[df["asset"] == asset]
    if rows.empty:
        return None
    return rows.sort_values(sort_col).iloc[-1]


# ── Signals ──────────────────────────────────────────────────────────────────

st.title("crypto-edge")
st.header("Signals")

signals_df = warehouse.read_table("predictions", "signals")
narrations_df = warehouse.read_table("predictions", "narrations")

cols = st.columns(2)
for col, asset in zip(cols, ASSETS):
    with col:
        st.subheader(ASSET_LABELS[asset])
        signal_row = _latest_for_asset(signals_df, asset, "predicted_at")
        if signal_row is None:
            st.info("No signal yet")
            continue
        badge = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}[signal_row["signal"]]
        st.metric(f"{badge} {signal_row['signal']}", f"{signal_row['confidence']:.0%} confidence")
        st.caption(_ago(signal_row["predicted_at"]))

        narration_row = _latest_for_asset(narrations_df, asset, "predicted_at")
        if narration_row is not None:
            st.write(narration_row["narration"])

        st.link_button(
            "Trade on Binance",
            f"https://www.binance.com/en/trade/{ASSET_LABELS[asset]}_USDT",
        )

# ── Charts ────────────────────────────────────────────────────────────────────

st.header("Charts")
ohlcv_df = warehouse.read_table("silver", "ohlcv")

for asset in ASSETS:
    st.subheader(ASSET_LABELS[asset])
    if ohlcv_df.empty or "asset" not in ohlcv_df.columns:
        st.info("No chart data yet")
        continue
    df = ohlcv_df[ohlcv_df["asset"] == asset].sort_values("open_time").copy()
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    df = df[df["open_time"] >= cutoff]
    if df.empty:
        st.info("No chart data yet")
        continue
    df["open_time"] = pd.to_datetime(df["open_time"])
    fig = go.Figure(data=[go.Candlestick(
        x=df["open_time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    )])

    if not signals_df.empty and "asset" in signals_df.columns:
        hist_df = signals_df[signals_df["asset"] == asset].copy()
        hist_df["predicted_at"] = pd.to_datetime(hist_df["predicted_at"])
        hist_df = hist_df[hist_df["predicted_at"] >= df["open_time"].min()]
        if not hist_df.empty:
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

# ── Feature Breakdown ─────────────────────────────────────────────────────────

st.header("Feature Breakdown")
for asset in ASSETS:
    signal_row = _latest_for_asset(signals_df, asset, "predicted_at")
    if signal_row is None:
        continue
    top5 = json.loads(signal_row["shap_top5"])
    if not top5:
        continue
    st.subheader(ASSET_LABELS[asset])
    bar_df = pd.DataFrame(top5)
    colors = ["green" if v > 0 else "red" for v in bar_df["value"]]
    fig = go.Figure(go.Bar(
        x=bar_df["value"], y=bar_df["feature"], orientation="h", marker_color=colors,
    ))
    st.plotly_chart(fig, use_container_width=True)

# ── Sentiment Feed ────────────────────────────────────────────────────────────

st.header("Sentiment Feed")
reddit_df = warehouse.read_table("silver", "reddit_posts")
news_df = warehouse.read_table("silver", "crypto_news")
feed_rows = []
if not reddit_df.empty:
    r = reddit_df.rename(columns={"subreddit": "source"})
    feed_rows.append(r[["source", "title", "url", "published_at", "sentiment", "confidence"]])
if not news_df.empty:
    feed_rows.append(
        news_df[["source", "title", "url", "published_at", "sentiment", "confidence"]]
    )
if feed_rows:
    combined = pd.concat(feed_rows, ignore_index=True).sort_values(
        "published_at", ascending=False
    )
    st.dataframe(combined.head(5)[["source", "sentiment", "title"]])
else:
    st.info("No sentiment data yet")

# ── Trade Journal ─────────────────────────────────────────────────────────────

st.header("Trade Journal")

with st.form("log_trade"):
    form_asset = st.selectbox("Asset", ASSETS, format_func=lambda a: ASSET_LABELS[a])
    direction = st.selectbox("Direction", ["LONG", "SHORT"])
    entry_price = st.number_input("Entry price", min_value=0.0)
    amount_usd = st.number_input("Amount (USD)", min_value=0.0)
    default_opened_at = datetime.now(timezone.utc).isoformat()
    opened_at = st.text_input("Opened at (ISO timestamp)", value=default_opened_at)
    submitted = st.form_submit_button("Log trade")
    if submitted:
        row = pd.DataFrame([{
            "id": str(uuid.uuid4()),
            "asset": form_asset,
            "direction": direction,
            "entry_price": entry_price,
            "amount_usd": amount_usd,
            "opened_at": pd.Timestamp(opened_at),
            "closed_at": pd.NaT,
            "exit_price": float("nan"),
        }])
        warehouse.write_table(row, "trades", "journal", mode="append")
        st.rerun()

st.subheader("Open positions")
journal_df = warehouse.read_table("trades", "journal")
if not journal_df.empty and "closed_at" in journal_df.columns:
    open_trades = journal_df[journal_df["closed_at"].isna()]
    if not open_trades.empty:
        current_ohlcv = warehouse.read_table("silver", "ohlcv")
        enriched = enrich_open_positions(open_trades.reset_index(drop=True), current_ohlcv)
        st.dataframe(
            enriched[["asset", "direction", "entry_price", "current_price", "unrealized_pnl"]]
        )
        for _, pos in enriched.iterrows():
            with st.expander(f"Close {ASSET_LABELS.get(pos['asset'], pos['asset'])} position"):
                exit_price = st.number_input(
                    "Exit price", min_value=0.0, key=f"exit_{pos['id']}"
                )
                if st.button("Close", key=f"close_{pos['id']}"):
                    idx = journal_df[journal_df["id"] == pos["id"]].index[0]
                    journal_df.loc[idx, "exit_price"] = exit_price
                    journal_df.loc[idx, "closed_at"] = pd.Timestamp.now().floor("s")
                    warehouse.write_table(journal_df, "trades", "journal", mode="replace")
                    st.rerun()
    else:
        st.info("No open positions")
else:
    st.info("No open positions")

st.subheader("Closed trades")
if not journal_df.empty and "closed_at" in journal_df.columns:
    closed = journal_df[journal_df["closed_at"].notna()]
    if not closed.empty:
        st.dataframe(closed[["asset", "direction", "entry_price", "exit_price"]])
    else:
        st.info("No closed trades yet")
else:
    st.info("No closed trades yet")

st.subheader("Performance summary")
if not journal_df.empty and "closed_at" in journal_df.columns:
    closed = journal_df[journal_df["closed_at"].notna()].copy()
    if not closed.empty:
        if not signals_df.empty:
            closed["matched_signal"] = closed.apply(
                lambda r: match_signal_for_trade(signals_df, r["asset"], r["opened_at"]),
                axis=1,
            )
        performance = summarize_performance(closed)
        perf_cols = st.columns(5)
        perf_cols[0].metric("Total P&L", f"${performance['total_pnl']:.2f}")
        win_rate = performance["win_rate"]
        perf_cols[1].metric("Win rate", f"{win_rate:.0%}" if win_rate is not None else "N/A")
        best = performance["best_trade_pnl"]
        worst = performance["worst_trade_pnl"]
        perf_cols[2].metric("Best trade", f"${best:.2f}" if best is not None else "N/A")
        perf_cols[3].metric("Worst trade", f"${worst:.2f}" if worst is not None else "N/A")
        sig_acc = performance["signal_accuracy"]
        perf_cols[4].metric(
            "Signal accuracy", f"{sig_acc:.0%}" if sig_acc is not None else "N/A"
        )
