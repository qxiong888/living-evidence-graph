# Deploy notes

Contest defaults:

- **Gemini 3.5 Flash** via free **Google AI Studio** key (`GEMINI_API_KEY`) — **Gemini API, not Vertex**
- Cloud Run **`us-central1`**, `--min-instances 0`, `--cpu-throttling`
- Cloud Scheduler **daily** → `POST /scheduler`
- Firestore Native optional (`LEG_USE_FIRESTORE=true`)

## Enable APIs

```bash
gcloud config set project "$GOOGLE_CLOUD_PROJECT"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  firestore.googleapis.com
```

Do **not** enable `aiplatform.googleapis.com` for the contest default path.

## Cloud Run

```bash
gcloud run deploy living-evidence-graph \
  --source . \
  --region us-central1 \
  --min-instances 0 \
  --cpu-throttling \
  --memory 512Mi \
  --max-instances 2 \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_GENAI_USE_VERTEXAI=false,GOOGLE_CLOUD_REGION=us-central1,GEMINI_MODEL=gemini-3.5-flash,LEG_USE_FIRESTORE=true,LEG_OUT_DIR=/tmp/leg-out,FIRESTORE_COLLECTION=evidence_graph" \
  --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY}"
```

## Cloud Scheduler (daily 09:00 America/Los_Angeles ≈ PT)

```bash
SERVICE_URL="$(gcloud run services describe living-evidence-graph --region us-central1 --format='value(status.url)')"

gcloud scheduler jobs create http leg-daily-keytruda \
  --location=us-central1 \
  --schedule="0 9 * * *" \
  --time-zone="America/Los_Angeles" \
  --uri="${SERVICE_URL}/scheduler" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{"goal":"pembrolizumab / Keytruda NSCLC solid tumor evidence graph"}'
```

See `scheduler.yaml`. Tear down after the demo video to avoid idle cost (min instances already 0).
