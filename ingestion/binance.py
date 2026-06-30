import pandas as pd


class BinanceClient:
    _BASE_URL = "https://api.binance.com/api/v3/klines"

    def fetch_ohlcv(self, symbol: str, interval: str = "4h") -> pd.DataFrame:
        import requests

        response = requests.get(
            self._BASE_URL,
            params={"symbol": symbol, "interval": interval, "limit": 1},
            timeout=10,
        )
        response.raise_for_status()
        rows = [
            {
                "asset": symbol,
                "open_time": pd.Timestamp(c[0], unit="ms", tz="UTC"),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
                "close_time": pd.Timestamp(c[6], unit="ms", tz="UTC"),
                "ingested_at": pd.Timestamp.now(tz="UTC"),
            }
            for c in response.json()
        ]
        return pd.DataFrame(rows)
