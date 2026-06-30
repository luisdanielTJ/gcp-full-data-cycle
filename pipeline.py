from adapters import get_llm, get_warehouse
from ingestion.run import run_ingestion_cycle
from silver.run import run_silver_cycle

if __name__ == "__main__":
    wh = get_warehouse()
    llm = get_llm()
    run_ingestion_cycle(wh)
    run_silver_cycle(wh, llm)
    print("Pipeline cycle complete")
