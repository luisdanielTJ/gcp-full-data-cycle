from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi.testclient import TestClient


def test_get_signal_returns_latest_row_for_asset():
    mock_warehouse = MagicMock()
    signals_df = pd.DataFrame([
        {"asset": "XBTUSD", "signal": "BUY", "confidence": 0.8,
         "predicted_at": pd.Timestamp("2026-06-01T00:00Z"), "model_version": "1",
         "shap_top5": "[]"},
        {"asset": "XBTUSD", "signal": "SELL", "confidence": 0.7,
         "predicted_at": pd.Timestamp("2026-06-01T04:00Z"), "model_version": "1",
         "shap_top5": "[]"},
    ])
    mock_warehouse.read_table.return_value = signals_df

    with patch("app.api.warehouse", mock_warehouse):
        from app.api import app
        client = TestClient(app)
        response = client.get("/signals/XBTUSD")

    assert response.status_code == 200
    assert response.json()["signal"] == "SELL"


def test_get_signal_returns_404_when_no_data():
    mock_warehouse = MagicMock()
    mock_warehouse.read_table.return_value = pd.DataFrame()

    with patch("app.api.warehouse", mock_warehouse):
        from app.api import app
        client = TestClient(app)
        response = client.get("/signals/XBTUSD")

    assert response.status_code == 404


def test_post_trade_inserts_row_with_generated_id():
    mock_warehouse = MagicMock()
    mock_warehouse.read_table.return_value = pd.DataFrame()

    with patch("app.api.warehouse", mock_warehouse):
        from app.api import app
        client = TestClient(app)
        response = client.post("/trades", json={
            "asset": "XBTUSD", "direction": "LONG", "entry_price": 50000.0,
            "amount_usd": 1000.0, "opened_at": "2026-06-01T00:00:00Z",
        })

    assert response.status_code == 200
    assert "id" in response.json()
    mock_warehouse.write_table.assert_called_once()
    written_df = mock_warehouse.write_table.call_args.args[0]
    assert written_df.iloc[0]["asset"] == "XBTUSD"
    assert pd.isna(written_df.iloc[0]["closed_at"])


def test_patch_close_trade_sets_exit_price_and_closed_at():
    mock_warehouse = MagicMock()
    journal_df = pd.DataFrame([{
        "id": "abc-123", "asset": "XBTUSD", "direction": "LONG", "entry_price": 50000.0,
        "amount_usd": 1000.0, "opened_at": pd.Timestamp("2026-06-01T00:00Z"),
        "closed_at": pd.NaT, "exit_price": float("nan"),
    }])
    mock_warehouse.read_table.return_value = journal_df

    with patch("app.api.warehouse", mock_warehouse):
        from app.api import app
        client = TestClient(app)
        response = client.patch("/trades/abc-123/close", json={"exit_price": 51000.0})

    assert response.status_code == 200
    mock_warehouse.write_table.assert_called_once()
    args, kwargs = mock_warehouse.write_table.call_args
    written_df = args[0]
    assert written_df.iloc[0]["exit_price"] == 51000.0
    assert pd.notna(written_df.iloc[0]["closed_at"])
    assert kwargs["mode"] == "replace"


def test_patch_close_trade_returns_404_when_trade_not_found():
    mock_warehouse = MagicMock()
    mock_warehouse.read_table.return_value = pd.DataFrame(columns=["id", "closed_at"])

    with patch("app.api.warehouse", mock_warehouse):
        from app.api import app
        client = TestClient(app)
        response = client.patch("/trades/nope/close", json={"exit_price": 1.0})

    assert response.status_code == 404


def test_get_positions_returns_enriched_open_trades():
    mock_warehouse = MagicMock()
    journal_df = pd.DataFrame([{
        "id": "t1", "asset": "XBTUSD", "direction": "LONG", "entry_price": 50000.0,
        "amount_usd": 1000.0, "opened_at": pd.Timestamp("2026-06-01T00:00Z"),
        "closed_at": pd.NaT, "exit_price": float("nan"),
    }])
    ohlcv_df = pd.DataFrame([{
        "asset": "XBTUSD", "open_time": pd.Timestamp("2026-06-01T04:00Z"), "close": 51000.0,
    }])
    mock_warehouse.read_table.side_effect = lambda dataset, table: (
        journal_df if table == "journal" else ohlcv_df if table == "ohlcv" else pd.DataFrame()
    )

    with patch("app.api.warehouse", mock_warehouse):
        from app.api import app
        client = TestClient(app)
        response = client.get("/positions")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["unrealized_pnl"] == 20.0


def test_get_performance_returns_summary_for_closed_trades():
    mock_warehouse = MagicMock()
    journal_df = pd.DataFrame([
        {"id": "t1", "asset": "XBTUSD", "direction": "LONG", "entry_price": 100.0,
         "exit_price": 110.0, "amount_usd": 1000.0,
         "opened_at": pd.Timestamp("2026-06-01T00:00Z"),
         "closed_at": pd.Timestamp("2026-06-01T04:00Z")},
    ])
    signals_df = pd.DataFrame([{
        "asset": "XBTUSD", "signal": "BUY", "predicted_at": pd.Timestamp("2026-05-31T00:00Z"),
    }])
    mock_warehouse.read_table.side_effect = lambda dataset, table: (
        journal_df if table == "journal" else signals_df if table == "signals" else pd.DataFrame()
    )

    with patch("app.api.warehouse", mock_warehouse):
        from app.api import app
        client = TestClient(app)
        response = client.get("/performance")

    assert response.status_code == 200
    body = response.json()
    assert body["total_pnl"] == 100.0
    assert body["win_rate"] == 1.0


def test_get_signal_history_returns_all_rows_for_asset():
    mock_warehouse = MagicMock()
    signals_df = pd.DataFrame([
        {"asset": "XBTUSD", "signal": "BUY", "confidence": 0.8,
         "predicted_at": pd.Timestamp("2026-06-01T00:00Z"), "model_version": "1", "shap_top5": "[]"},
        {"asset": "ETHUSD", "signal": "SELL", "confidence": 0.7,
         "predicted_at": pd.Timestamp("2026-06-01T00:00Z"), "model_version": "1", "shap_top5": "[]"},
    ])
    mock_warehouse.read_table.return_value = signals_df

    with patch("app.api.warehouse", mock_warehouse):
        from app.api import app
        client = TestClient(app)
        response = client.get("/signals/XBTUSD/history")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["signal"] == "BUY"


def test_get_sentiment_combines_reddit_and_news():
    mock_warehouse = MagicMock()
    reddit_df = pd.DataFrame([{
        "subreddit": "Bitcoin", "title": "BTC moon", "url": "https://a",
        "published_at": pd.Timestamp("2026-06-01T00:00Z"), "sentiment": 1, "confidence": 0.9,
    }])
    news_df = pd.DataFrame([{
        "source": "coindesk", "title": "BTC dips", "url": "https://b",
        "published_at": pd.Timestamp("2026-06-01T01:00Z"), "sentiment": -1, "confidence": 0.8,
    }])
    mock_warehouse.read_table.side_effect = lambda dataset, table: (
        reddit_df if table == "reddit_posts" else news_df if table == "crypto_news" else pd.DataFrame()
    )

    with patch("app.api.warehouse", mock_warehouse):
        from app.api import app
        client = TestClient(app)
        response = client.get("/sentiment")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    sources = {row["source"] for row in body}
    assert sources == {"Bitcoin", "coindesk"}
