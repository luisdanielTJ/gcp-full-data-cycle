import pandas as pd


class KrakenClient:
    _BASE_URL = "https://api.kraken.com/0/public/OHLC"
    _INTERVAL_MINUTES = 240  # 4 hours

    def fetch_ohlcv(self, pair: str) -> pd.DataFrame:
        import requests

        response = requests.get(
            self._BASE_URL,
            params={"pair": pair, "interval": self._INTERVAL_MINUTES},
            timeout=10,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("error"):
            raise ValueError(f"Kraken API error: {body['error']}")
        result = body["result"]
        candles = next(v for k, v in result.items() if k != "last")
        c = candles[-1]
        return pd.DataFrame(
            [
                {
                    "asset": pair,
                    "open_time": pd.Timestamp(c[0], unit="s", tz="UTC"),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[6]),
                    "ingested_at": pd.Timestamp.now(tz="UTC"),
                }
            ]
        )
