import json
import uuid
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adapters import get_warehouse
from app.pnl import match_signal_for_trade, summarize_performance
from app.positions import enrich_open_positions

st.set_page_config(page_title="crypto-edge", page_icon="📈", layout="wide")

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


def _styled_chart(ohlcv_df: pd.DataFrame, asset: str, signals_df: pd.DataFrame) -> go.Figure | None:
    df = ohlcv_df[ohlcv_df["asset"] == asset].sort_values("open_time").copy()
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    df = df[df["open_time"] >= cutoff]
    if df.empty:
        return None
    df["open_time"] = pd.to_datetime(df["open_time"])
    fig = go.Figure(data=[go.Candlestick(
        x=df["open_time"],
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name="Price",
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
            for signal_type, color, symbol in (
                ("BUY", "#26a69a", "triangle-up"),
                ("SELL", "#ef5350", "triangle-down"),
            ):
                points = merged[merged["signal"] == signal_type]
                if not points.empty:
                    fig.add_trace(go.Scatter(
                        x=points["open_time"], y=points["close"], mode="markers",
                        marker=dict(color=color, size=12, symbol=symbol),
                        name=signal_type,
                    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#eaf4fb",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=8, b=0),
        height=320,
        xaxis=dict(
            gridcolor="rgba(180,210,230,0.6)",
            showline=True,
            linecolor="rgba(100,160,200,0.5)",
        ),
        yaxis=dict(
            gridcolor="rgba(180,210,230,0.6)",
            showline=True,
            linecolor="rgba(100,160,200,0.5)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Load data ─────────────────────────────────────────────────────────────────

signals_df = warehouse.read_table("predictions", "signals")
narrations_df = warehouse.read_table("predictions", "narrations")
ohlcv_df = warehouse.read_table("silver", "ohlcv")

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📈 crypto-edge")

# ── Signal cards ──────────────────────────────────────────────────────────────

sig_cols = st.columns(2)
for col, asset in zip(sig_cols, ASSETS):
    with col:
        with st.container(border=True):
            signal_row = _latest_for_asset(signals_df, asset, "predicted_at")
            if signal_row is None:
                st.subheader(ASSET_LABELS[asset])
                st.info("No signal yet")
                continue
            badge = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}[signal_row["signal"]]
            st.subheader(f"{ASSET_LABELS[asset]}  {badge} {signal_row['signal']}")
            m_col, t_col = st.columns([1, 1])
            m_col.metric("Confidence", f"{signal_row['confidence']:.0%}")
            t_col.caption(f"\n{_ago(signal_row['predicted_at'])}")
            narration_row = _latest_for_asset(narrations_df, asset, "predicted_at")
            if narration_row is not None:
                with st.expander("View analysis"):
                    st.write(narration_row["narration"])
            st.link_button(
                f"Trade {ASSET_LABELS[asset]} on Binance",
                f"https://www.binance.com/en/trade/{ASSET_LABELS[asset]}_USDT",
            )

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_charts, tab_analysis, tab_sentiment, tab_paper, tab_journal = st.tabs([
    "📈 Charts",
    "🔍 Feature Analysis",
    "📰 Sentiment",
    "🤖 Paper Trading",
    "📒 Trade Journal",
])

# ── Charts tab ────────────────────────────────────────────────────────────────

with tab_charts:
    chart_cols = st.columns(2)
    for col, asset in zip(chart_cols, ASSETS):
        with col:
            with st.container(border=True):
                st.subheader(ASSET_LABELS[asset])
                if ohlcv_df.empty or "asset" not in ohlcv_df.columns:
                    st.info("No chart data yet")
                else:
                    fig = _styled_chart(ohlcv_df, asset, signals_df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No chart data yet")

# ── Feature Analysis tab ──────────────────────────────────────────────────────

with tab_analysis:
    analysis_cols = st.columns(2)
    for col, asset in zip(analysis_cols, ASSETS):
        with col:
            with st.container(border=True):
                st.subheader(f"{ASSET_LABELS[asset]} — Top drivers")
                signal_row = _latest_for_asset(signals_df, asset, "predicted_at")
                if signal_row is None:
                    st.info("No signal data yet")
                    continue
                top5 = json.loads(signal_row["shap_top5"])
                if not top5:
                    st.info("No feature data yet")
                    continue
                bar_df = pd.DataFrame(top5)
                colors = ["#26a69a" if v > 0 else "#ef5350" for v in bar_df["value"]]
                fig = go.Figure(go.Bar(
                    x=bar_df["value"], y=bar_df["feature"], orientation="h",
                    marker_color=colors,
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#eaf4fb",
                    margin=dict(l=0, r=0, t=8, b=0),
                    height=240,
                    xaxis=dict(gridcolor="rgba(180,210,230,0.6)"),
                    yaxis=dict(gridcolor="rgba(180,210,230,0.6)"),
                )
                st.plotly_chart(fig, use_container_width=True)

# ── Sentiment tab ─────────────────────────────────────────────────────────────

with tab_sentiment:
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
        with st.container(border=True):
            st.dataframe(
                combined.head(10)[["source", "sentiment", "title"]],
                use_container_width=True,
            )
    else:
        st.info("No sentiment data yet")

# ── Paper Trading tab ─────────────────────────────────────────────────────────

with tab_paper:
    paper_df = warehouse.read_table("paper_trades", "journal")

    closed_paper = (
        paper_df[paper_df["closed_at"].notna()] if not paper_df.empty else pd.DataFrame()
    )
    open_paper = (
        paper_df[paper_df["closed_at"].isna()] if not paper_df.empty else pd.DataFrame()
    )

    total_pnl = float(closed_paper["pnl_usd"].sum()) if not closed_paper.empty else 0.0
    paper_win_rate = (
        float((closed_paper["pnl_usd"] > 0).mean()) if not closed_paper.empty else None
    )

    pt_cols = st.columns(4)
    pt_cols[0].metric("Total P&L", f"${total_pnl:+.2f}")
    pt_cols[1].metric("Win Rate", f"{paper_win_rate:.0%}" if paper_win_rate is not None else "N/A")
    pt_cols[2].metric("Open Positions", len(open_paper))
    pt_cols[3].metric("Total Trades", len(paper_df) if not paper_df.empty else 0)

    st.subheader("Open positions")
    if not open_paper.empty:
        open_rows = []
        for _, pos in open_paper.iterrows():
            asset = pos["asset"]
            entry = float(pos["entry_price"])
            ohlcv_row = _latest_for_asset(ohlcv_df, asset, "open_time")
            if ohlcv_row is None:
                continue
            current = float(ohlcv_row["close"])
            upnl_pct = (current - entry) / entry
            upnl_usd = upnl_pct * float(pos["position_size_usd"])
            open_rows.append({
                "Asset": ASSET_LABELS.get(asset, asset),
                "Entry ($)": f"{entry:,.2f}",
                "Current ($)": f"{current:,.2f}",
                "Unrealized P&L ($)": f"{upnl_usd:+.2f}",
                "Unrealized P&L (%)": f"{upnl_pct:+.1%}",
                "Opened At": pos["opened_at"],
            })
        if open_rows:
            with st.container(border=True):
                st.dataframe(pd.DataFrame(open_rows), use_container_width=True)
        else:
            st.info("No price data for open positions")
    else:
        st.info("No open positions yet — waiting for a BUY signal ≥ 70% confidence")

    st.subheader("Closed trades")
    if not closed_paper.empty:
        display = closed_paper[
            ["asset", "entry_price", "exit_price", "pnl_usd", "pnl_pct", "opened_at", "closed_at"]
        ].copy()
        display["asset"] = display["asset"].map(ASSET_LABELS).fillna(display["asset"])
        display["pnl_pct"] = display["pnl_pct"].apply(
            lambda x: f"{x:+.1%}" if pd.notna(x) else ""
        )
        display["pnl_usd"] = display["pnl_usd"].apply(
            lambda x: f"${x:+.2f}" if pd.notna(x) else ""
        )
        display = display.rename(columns={
            "asset": "Asset",
            "entry_price": "Entry ($)",
            "exit_price": "Exit ($)",
            "pnl_usd": "P&L ($)",
            "pnl_pct": "P&L (%)",
            "opened_at": "Opened At",
            "closed_at": "Closed At",
        })
        with st.container(border=True):
            st.dataframe(
                display.sort_values("Closed At", ascending=False),
                use_container_width=True,
            )
    else:
        st.info("No closed trades yet")

# ── Trade Journal tab ─────────────────────────────────────────────────────────

with tab_journal:
    journal_df = warehouse.read_table("trades", "journal")

    with st.container(border=True):
        st.subheader("Log a trade")
        with st.form("log_trade"):
            form_cols = st.columns(3)
            form_asset = form_cols[0].selectbox("Asset", ASSETS, format_func=lambda a: ASSET_LABELS[a])
            direction = form_cols[1].selectbox("Direction", ["LONG", "SHORT"])
            entry_price = form_cols[2].number_input("Entry price", min_value=0.0)
            form_cols2 = st.columns(2)
            amount_usd = form_cols2[0].number_input("Amount (USD)", min_value=0.0)
            default_opened_at = datetime.now(timezone.utc).isoformat()
            opened_at = form_cols2[1].text_input("Opened at (ISO timestamp)", value=default_opened_at)
            submitted = st.form_submit_button("Log trade", use_container_width=True)
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

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("Open positions")
            if not journal_df.empty and "closed_at" in journal_df.columns:
                open_trades = journal_df[journal_df["closed_at"].isna()]
                if not open_trades.empty:
                    current_ohlcv = ohlcv_df
                    enriched = enrich_open_positions(open_trades.reset_index(drop=True), current_ohlcv)
                    st.dataframe(
                        enriched[["asset", "direction", "entry_price", "current_price", "unrealized_pnl"]],
                        use_container_width=True,
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

    with right:
        with st.container(border=True):
            st.subheader("Closed trades")
            if not journal_df.empty and "closed_at" in journal_df.columns:
                closed = journal_df[journal_df["closed_at"].notna()]
                if not closed.empty:
                    st.dataframe(
                        closed[["asset", "direction", "entry_price", "exit_price"]],
                        use_container_width=True,
                    )
                else:
                    st.info("No closed trades yet")
            else:
                st.info("No closed trades yet")

    with st.container(border=True):
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
            else:
                st.info("No closed trades yet")
        else:
            st.info("No closed trades yet")
