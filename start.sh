#!/bin/sh
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then
  echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /tmp/gcp-credentials.json
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-credentials.json
fi
uvicorn app.api:app --host 0.0.0.0 --port 8000 &
streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0
