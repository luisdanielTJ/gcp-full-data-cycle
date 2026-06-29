import os
from dotenv import load_dotenv

load_dotenv()

GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
GCP_REGION: str = os.getenv("GCP_REGION", "us-central1")
WAREHOUSE_MODE: str = os.getenv("WAREHOUSE_MODE", "duckdb")
LLM_MODE: str = os.getenv("LLM_MODE", "gemini")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
DUCKDB_PATH: str = os.getenv("DUCKDB_PATH", ":memory:")
