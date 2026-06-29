import pytest

from adapters.model_registry import InMemoryModelRegistry


@pytest.fixture
def registry():
    return InMemoryModelRegistry()


def test_log_returns_version_string(registry):
    version = registry.log_model(
        model={"type": "xgboost"},
        metrics={"signal_accuracy": 0.62},
        params={"n_estimators": 100},
        name="btc_signal",
    )
    assert isinstance(version, str)
    assert len(version) > 0


def test_log_and_load_by_version(registry):
    model = {"type": "xgboost", "n_estimators": 100}
    version = registry.log_model(model=model, metrics={}, params={}, name="btc_signal")
    loaded = registry.load_model("btc_signal", version=version)
    assert loaded == model


def test_promote_and_load_production(registry):
    model = {"type": "xgboost"}
    version = registry.log_model(model=model, metrics={}, params={}, name="btc_signal")
    registry.promote_model("btc_signal", version)
    loaded = registry.load_model("btc_signal", version="production")
    assert loaded == model


def test_load_production_without_promotion_raises(registry):
    with pytest.raises(ValueError, match="No production model"):
        registry.load_model("btc_signal", version="production")


def test_versions_increment(registry):
    v1 = registry.log_model(model={"v": 1}, metrics={}, params={}, name="btc_signal")
    v2 = registry.log_model(model={"v": 2}, metrics={}, params={}, name="btc_signal")
    assert v1 != v2
    assert registry.load_model("btc_signal", version=v1) == {"v": 1}
    assert registry.load_model("btc_signal", version=v2) == {"v": 2}
