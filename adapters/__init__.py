from adapters.config import (
    DATABASE_URL,
    DUCKDB_PATH,
    GCP_PROJECT_ID,
    GEMINI_API_KEY,
    LLM_MODE,
    OPENAI_API_KEY,
    WAREHOUSE_MODE,
)
from adapters.llm import GeminiAdapter, LLMAdapter, OpenAIAdapter
from adapters.model_registry import ModelRegistryAdapter, WarehouseModelRegistry
from adapters.warehouse import (
    BigQueryWarehouse,
    DuckDBWarehouse,
    SupabaseWarehouseAdapter,
    WarehouseAdapter,
)


def get_warehouse() -> WarehouseAdapter:
    if WAREHOUSE_MODE == "supabase":
        return SupabaseWarehouseAdapter(database_url=DATABASE_URL)
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
    return WarehouseModelRegistry(get_warehouse())
