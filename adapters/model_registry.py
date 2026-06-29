from abc import ABC, abstractmethod
from typing import Any


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
