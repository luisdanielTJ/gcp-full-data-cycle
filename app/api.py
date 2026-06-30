import math
import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from adapters import get_warehouse
from app.pnl import match_signal_for_trade, summarize_performance
from app.positions import enrich_open_positions

app = FastAPI(title="crypto-edge API")
warehouse = get_warehouse()


class TradeCreate(BaseModel):
    asset: str
    direction: str
    entry_price: float
    amount_usd: float
    opened_at: str


class TradeClose(BaseModel):
    exit_price: float


def _records(df: pd.DataFrame) -> list[dict]:
    raw = df.where(pd.notna(df), None).to_dict(orient="records")
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in rec.items()}
        for rec in raw
    ]


@app.get("/signals/{asset}")
def get_signal(asset: str):
    df = warehouse.read_table("predictions", "signals")
    if df.empty or "asset" not in df.columns:
        raise HTTPException(status_code=404, detail=f"No signal for {asset}")
    df = df[df["asset"] == asset]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No signal for {asset}")
    latest = df.sort_values("predicted_at").iloc[-1]
    return _records(latest.to_frame().T)[0]


@app.get("/narration/{asset}")
def get_narration(asset: str):
    df = warehouse.read_table("predictions", "narrations")
    df = df[df["asset"] == asset]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No narration for {asset}")
    latest = df.sort_values("predicted_at").iloc[-1]
    return _records(latest.to_frame().T)[0]


@app.get("/ohlcv/{asset}")
def get_ohlcv(asset: str):
    df = warehouse.read_table("silver", "ohlcv")
    df = df[df["asset"] == asset].sort_values("open_time")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    df = df[df["open_time"] >= cutoff]
    return _records(df)


@app.get("/sentiment")
def get_sentiment():
    reddit_df = warehouse.read_table("silver", "reddit_posts")
    news_df = warehouse.read_table("silver", "crypto_news")

    rows = []
    if not reddit_df.empty:
        r = reddit_df.rename(columns={"subreddit": "source"})
        rows.append(r[["source", "title", "url", "published_at", "sentiment", "confidence"]])
    if not news_df.empty:
        n = news_df[["source", "title", "url", "published_at", "sentiment", "confidence"]]
        rows.append(n)

    if not rows:
        return []

    combined = pd.concat(rows, ignore_index=True).sort_values("published_at", ascending=False)
    return _records(combined.head(5))


@app.get("/trades")
def get_trades():
    df = warehouse.read_table("trades", "journal")
    return _records(df)


@app.post("/trades")
def post_trade(trade: TradeCreate):
    row = pd.DataFrame([{
        "id": str(uuid.uuid4()),
        "asset": trade.asset,
        "direction": trade.direction,
        "entry_price": trade.entry_price,
        "amount_usd": trade.amount_usd,
        "opened_at": pd.Timestamp(trade.opened_at),
        "closed_at": pd.NaT,
        "exit_price": float("nan"),
    }])
    warehouse.write_table(row, "trades", "journal", mode="append")
    return _records(row)[0]


@app.get("/positions")
def get_positions():
    journal_df = warehouse.read_table("trades", "journal")
    if journal_df.empty or "closed_at" not in journal_df.columns:
        return []
    open_trades = journal_df[journal_df["closed_at"].isna()]
    ohlcv_df = warehouse.read_table("silver", "ohlcv")
    enriched = enrich_open_positions(open_trades.reset_index(drop=True), ohlcv_df)
    return _records(enriched)


@app.get("/performance")
def get_performance():
    journal_df = warehouse.read_table("trades", "journal")
    if journal_df.empty or "closed_at" not in journal_df.columns:
        return summarize_performance(pd.DataFrame())
    closed = journal_df[journal_df["closed_at"].notna()].copy()
    if closed.empty:
        return summarize_performance(pd.DataFrame())
    signals_df = warehouse.read_table("predictions", "signals")
    closed["matched_signal"] = closed.apply(
        lambda r: match_signal_for_trade(signals_df, r["asset"], r["opened_at"])
        if not signals_df.empty else None,
        axis=1,
    )
    return summarize_performance(closed)


@app.get("/signals/{asset}/history")
def get_signal_history(asset: str):
    df = warehouse.read_table("predictions", "signals")
    if df.empty or "asset" not in df.columns:
        return []
    df = df[df["asset"] == asset].sort_values("predicted_at")
    return _records(df)


@app.patch("/trades/{trade_id}/close")
def patch_close_trade(trade_id: str, close: TradeClose):
    df = warehouse.read_table("trades", "journal")
    if df.empty or trade_id not in set(df["id"]):
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

    idx = df[df["id"] == trade_id].index[0]
    df.loc[idx, "exit_price"] = close.exit_price
    df.loc[idx, "closed_at"] = pd.Timestamp.now().floor("s")

    warehouse.write_table(df, "trades", "journal", mode="replace")
    return _records(df.loc[[idx]])[0]
