import os

from dotenv import load_dotenv

load_dotenv()

WAREHOUSE_MODE: str = os.getenv("WAREHOUSE_MODE", "duckdb")
LLM_MODE: str = os.getenv("LLM_MODE", "openai")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "crypto-edge-ingestion/0.1")
DUCKDB_PATH: str = os.getenv("DUCKDB_PATH", ":memory:")
DATABASE_URL: str = os.getenv("DATABASE_URL", "")
