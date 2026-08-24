#!/usr/bin/env bash
# Local spin-up notes — not a GCP deploy.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_GENAI_USE_VERTEXAI=false
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.5-flash}"
pytest tests/test_credibility.py -q
python scripts/demo_local.py
echo "Demo artifacts under out/demo/"
echo "API: uvicorn living_evidence_graph.server:app --host 0.0.0.0 --port 8080"
