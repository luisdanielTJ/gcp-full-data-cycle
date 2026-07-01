from adapters.config import (
    DATABASE_URL,
    DUCKDB_PATH,
    LLM_MODE,
    OPENAI_API_KEY,
    WAREHOUSE_MODE,
)
from adapters.llm import LLMAdapter, OpenAIAdapter
from adapters.model_registry import ModelRegistryAdapter, WarehouseModelRegistry
from adapters.warehouse import DuckDBWarehouse, SupabaseWarehouseAdapter, WarehouseAdapter


def get_warehouse() -> WarehouseAdapter:
    if WAREHOUSE_MODE == "supabase":
        return SupabaseWarehouseAdapter(database_url=DATABASE_URL)
    return DuckDBWarehouse(db_path=DUCKDB_PATH)


def get_llm() -> LLMAdapter:
    if LLM_MODE == "openai":
        return OpenAIAdapter(api_key=OPENAI_API_KEY)
    raise ValueError(f"Unknown LLM_MODE: {LLM_MODE!r}. Set LLM_MODE=openai in .env")


def get_model_registry() -> ModelRegistryAdapter:
    return WarehouseModelRegistry(get_warehouse())
