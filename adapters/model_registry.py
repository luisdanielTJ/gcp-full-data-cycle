import json
import pickle
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class ModelRegistryAdapter(ABC):
    @abstractmethod
    def log_model(self, model: Any, metrics: dict, params: dict, name: str) -> str:
        """Store model artifact and metadata. Returns a version string."""
        ...

    @abstractmethod
    def load_model(self, name: str, version: str = "production") -> Any:
        ...

    @abstractmethod
    def promote_model(self, name: str, version: str) -> None:
        ...

    @abstractmethod
    def get_production_version(self, name: str) -> str | None:
        """Returns the currently promoted version, or None if unpromoted."""
        ...


class InMemoryModelRegistry(ModelRegistryAdapter):
    """Local stub for testing. Vertex AI implementation added in Plan 3."""

    def __init__(self) -> None:
        self._models: dict[str, dict[str, dict]] = {}
        self._production: dict[str, str] = {}

    def log_model(self, model: Any, metrics: dict, params: dict, name: str) -> str:
        existing = self._models.get(name, {})
        version = str(len(existing) + 1)
        self._models.setdefault(name, {})[version] = {
            "model": model,
            "metrics": metrics,
            "params": params,
        }
        return version

    def load_model(self, name: str, version: str = "production") -> Any:
        if version == "production":
            version = self._production.get(name, "")
            if not version:
                raise ValueError(f"No production model for '{name}'")
        return self._models[name][version]["model"]

    def promote_model(self, name: str, version: str) -> None:
        self._production[name] = version

    def get_production_version(self, name: str) -> str | None:
        return self._production.get(name)


class WarehouseModelRegistry(ModelRegistryAdapter):
    """Stores models in warehouse tables (models.registry, models.promotions)."""

    def __init__(self, warehouse) -> None:
        self.warehouse = warehouse

    def log_model(self, model: Any, metrics: dict, params: dict, name: str) -> str:
        registry_df = self.warehouse.read_table("models", "registry")
        existing_count = 0 if registry_df.empty else len(registry_df[registry_df["name"] == name])
        version = str(existing_count + 1)

        row = pd.DataFrame([{
            "name": name,
            "version": version,
            "model_bytes": pickle.dumps(model),
            "metrics_json": json.dumps(metrics),
            "params_json": json.dumps(params),
            "created_at": pd.Timestamp.now(tz="UTC"),
        }])
        self.warehouse.write_table(row, "models", "registry", mode="append")
        return version

    def load_model(self, name: str, version: str = "production") -> Any:
        if version == "production":
            promotions_df = self.warehouse.read_table("models", "promotions")
            if not promotions_df.empty:
                promotions_df = promotions_df[promotions_df["name"] == name]
            if promotions_df.empty:
                raise ValueError(f"No production model for '{name}'")
            version = promotions_df.sort_values("promoted_at").iloc[-1]["version"]

        registry_df = self.warehouse.read_table("models", "registry")
        match = registry_df[(registry_df["name"] == name) & (registry_df["version"] == version)]
        return pickle.loads(match.iloc[0]["model_bytes"])

    def promote_model(self, name: str, version: str) -> None:
        row = pd.DataFrame([{
            "name": name,
            "version": version,
            "promoted_at": pd.Timestamp.now(tz="UTC"),
        }])
        self.warehouse.write_table(row, "models", "promotions", mode="append")

    def get_production_version(self, name: str) -> str | None:
        promotions_df = self.warehouse.read_table("models", "promotions")
        if not promotions_df.empty:
            promotions_df = promotions_df[promotions_df["name"] == name]
        if promotions_df.empty:
            return None
        return promotions_df.sort_values("promoted_at").iloc[-1]["version"]
