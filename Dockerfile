# Cloud Run (us-central1). Contest default: Gemini 3.5 Flash via Gemini API
# (AI Studio key). Not Vertex. min-instances 0 set at deploy time.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    LEG_OUT_DIR=/tmp/leg-out \
    GOOGLE_GENAI_USE_VERTEXAI=false \
    GEMINI_MODEL=gemini-3.5-flash \
    GOOGLE_CLOUD_REGION=us-central1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY living_evidence_graph ./living_evidence_graph
# Explicit static copy (hub /compare /update /push). Do not COPY out/ or scripts/.
COPY living_evidence_graph/static ./living_evidence_graph/static
# fixtures/ includes demo_graph/ (baked Keytruda/NSCLC 14/10 JSON).
COPY fixtures ./fixtures

EXPOSE 8080
CMD ["uvicorn", "living_evidence_graph.server:app", "--host", "0.0.0.0", "--port", "8080"]
