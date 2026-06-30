from adapters import get_llm, get_model_registry, get_warehouse
from ml.predict import run_prediction_cycle

if __name__ == "__main__":
    wh = get_warehouse()
    registry = get_model_registry()
    llm = get_llm()
    run_prediction_cycle(wh, registry, llm)
    print("Prediction cycle complete")
