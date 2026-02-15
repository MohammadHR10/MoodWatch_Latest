# GCP Deployment Guide

Deploy the Multimodal Emotion Analyzer to Google Cloud Platform.

## Prerequisites

1. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
2. A GCP project with billing enabled
3. Docker installed (for local testing)

## Quick Deploy to Cloud Run (Recommended)

Cloud Run is serverless and scales to zero when not in use.

```bash
# 1. Set your project ID
export PROJECT_ID=your-project-id
gcloud config set project $PROJECT_ID

# 2. Enable required APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com

# 3. Build and deploy
gcloud builds submit --config cloudbuild.yaml

# Or deploy manually:
gcloud run deploy emotion-analyzer \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "EMOTION_BACKEND=pyfeat,GROQ_API_KEY=your-key"
```

## Environment Variables

Set these in Cloud Run or App Engine:

| Variable          | Required        | Description                          |
| ----------------- | --------------- | ------------------------------------ |
| `GROQ_API_KEY`    | Yes (for audio) | Groq API key for audio transcription |
| `EMOTION_BACKEND` | No              | `pyfeat` (default for cloud)         |
| `SECRET_KEY`      | Recommended     | Flask session secret                 |

### Set secrets securely:

```bash
# Using Cloud Run secrets
gcloud run services update emotion-analyzer \
  --set-env-vars "GROQ_API_KEY=your-groq-api-key"
```

## Alternative: App Engine

```bash
# Deploy to App Engine Standard
gcloud app deploy app.yaml

# View logs
gcloud app logs tail -s default
```

## Local Testing with Docker

```bash
# Build
docker build -t emotion-analyzer .

# Run
docker run -p 8080:8080 \
  -e GROQ_API_KEY=your-key \
  -e EMOTION_BACKEND=pyfeat \
  emotion-analyzer

# Open http://localhost:8080
```

## Architecture Notes

- **Video Analysis**: Uses py-feat for facial Action Unit detection (no OpenFace in cloud)
- **Audio Analysis**: Uses Groq API (Whisper + LLaMA) for transcription and emotion analysis
- **Storage**: Uploads and sessions are stored in-memory/temp (add Cloud Storage for persistence)

## Scaling Configuration

Cloud Run auto-scales. Adjust in `cloudbuild.yaml`:

```yaml
--min-instances 0      # Scale to zero
--max-instances 10     # Max concurrent instances
--memory 2Gi           # Memory per instance
--cpu 2                # CPUs per instance
```

## Costs

- **Cloud Run**: Pay only for requests (free tier: 2M requests/month)
- **Cloud Build**: 120 free build-minutes/day
- **Groq API**: Free tier available at https://console.groq.com

## Troubleshooting

### Build fails

```bash
# Check logs
gcloud builds log [BUILD_ID]
```

### Container crashes

```bash
# View Cloud Run logs
gcloud run services logs read emotion-analyzer --region us-central1
```

### Video analysis slow

- py-feat downloads models on first use (~500MB)
- First request may timeout; increase `--timeout` if needed
