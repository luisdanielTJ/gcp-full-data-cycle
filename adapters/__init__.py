from adapters.config import (
    WAREHOUSE_MODE,
    LLM_MODE,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    DUCKDB_PATH,
    GCP_PROJECT_ID,
)
from adapters.warehouse import WarehouseAdapter, DuckDBWarehouse, BigQueryWarehouse
from adapters.llm import LLMAdapter, GeminiAdapter, OpenAIAdapter
from adapters.model_registry import ModelRegistryAdapter, InMemoryModelRegistry


def get_warehouse() -> WarehouseAdapter:
    if WAREHOUSE_MODE == "bigquery":
        return BigQueryWarehouse(project_id=GCP_PROJECT_ID)
    return DuckDBWarehouse(db_path=DUCKDB_PATH)


def get_llm() -> LLMAdapter:
    if LLM_MODE == "gemini":
        return GeminiAdapter(api_key=GEMINI_API_KEY)
    if LLM_MODE == "openai":
        return OpenAIAdapter(api_key=OPENAI_API_KEY)
    raise ValueError(f"Unknown LLM_MODE: {LLM_MODE!r}. Set LLM_MODE=gemini or openai in .env")


def get_model_registry() -> ModelRegistryAdapter:
    return InMemoryModelRegistry()
