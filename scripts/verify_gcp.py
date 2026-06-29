"""Run once after GCP console setup to verify all connections work."""
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def verify_bigquery() -> None:
    from adapters.warehouse import BigQueryWarehouse
    project_id = os.environ["GCP_PROJECT_ID"]
    print(f"Testing BigQuery -> project: {project_id}")
    wh = BigQueryWarehouse(project_id=project_id)
    result = wh.run_query("SELECT 1 AS ping")
    assert result["ping"].iloc[0] == 1
    print("[OK] BigQuery OK")


def verify_llm() -> None:
    from adapters import get_llm
    mode = os.getenv("LLM_MODE", "gemini")
    print(f"Testing {mode} LLM adapter...")
    adapter = get_llm()
    result = adapter.score_sentiment("Bitcoin ETF approved")
    assert result["sentiment"] in (-1, 0, 1), f"Unexpected sentiment: {result}"
    print(f"[OK] {mode} OK - test sentiment: {result['sentiment']} ({result['reason']})")


if __name__ == "__main__":
    try:
        verify_bigquery()
        verify_llm()
        print("\n[OK] All GCP connections verified")
    except Exception as exc:
        print(f"\n[FAIL] Failed: {exc}")
        print("\nTroubleshooting:")
        print("  BigQuery auth error -> run: gcloud auth application-default login")
        print("  LLM error -> check OPENAI_API_KEY or GEMINI_API_KEY matches LLM_MODE in .env")
        sys.exit(1)
