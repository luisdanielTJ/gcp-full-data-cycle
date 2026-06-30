import logging

import pandas as pd

logger = logging.getLogger(__name__)


def score_new_posts(bronze_df: pd.DataFrame, existing_urls: set[str], llm) -> pd.DataFrame:
    if bronze_df.empty:
        return bronze_df

    new_rows = bronze_df[~bronze_df["url"].isin(existing_urls)]

    scored = []
    for _, row in new_rows.iterrows():
        try:
            result = llm.score_sentiment(row["title"])
        except Exception:
            logger.warning("Sentiment scoring failed for %s", row["url"], exc_info=True)
            continue
        scored_row = row.to_dict()
        scored_row["sentiment"] = result["sentiment"]
        scored_row["confidence"] = result["confidence"]
        scored_row["reason"] = result["reason"]
        scored_row["scored_at"] = pd.Timestamp.now(tz="UTC")
        scored.append(scored_row)

    return pd.DataFrame(scored)
