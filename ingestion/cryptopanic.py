import pandas as pd


class CryptoPanicClient:
    _BASE_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def fetch_news(self, currencies: list[str]) -> pd.DataFrame:
        import requests

        rows = []
        for currency in currencies:
            response = requests.get(
                self._BASE_URL,
                params={"auth_token": self._api_key, "currencies": currency, "public": "true"},
                timeout=10,
            )
            response.raise_for_status()
            for item in response.json().get("results", []):
                rows.append(
                    {
                        "currency": currency,
                        "title": item["title"],
                        "published_at": pd.Timestamp(item["published_at"]).tz_convert("UTC"),
                        "url": item["url"],
                        "ingested_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
        return pd.DataFrame(rows)
