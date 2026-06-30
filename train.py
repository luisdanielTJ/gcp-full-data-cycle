from adapters import get_model_registry, get_warehouse
from ml.run import run_training_cycle

if __name__ == "__main__":
    wh = get_warehouse()
    registry = get_model_registry()
    run_training_cycle(wh, registry)
    print("Training cycle complete")
