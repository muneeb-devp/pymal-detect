# CI/CD Configuration for Google Cloud Run

## Required GitHub Secrets

To enable automatic deployment via GitHub Actions, configure these secrets in your repository:

### 1. GCP_SA_KEY

**Description:** Google Cloud Service Account Key (JSON format)

**How to Create:**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **IAM & Admin** > **Service Accounts**
3. Click **Create Service Account**
4. Name: `github-actions-deploy`
5. Grant roles:
   - Cloud Run Admin
   - Service Account User
   - Storage Admin (for Cloud Build)
6. Click **Create Key** → Choose **JSON**
7. Save the JSON file
8. In GitHub: **Settings** > **Secrets and variables** > **Actions** > **New repository secret**
9. Name: `GCP_SA_KEY`
10. Value: Paste the entire JSON content

### 2. GCP_PROJECT_ID

**Description:** Your Google Cloud Project ID

**How to Find:**
1. Go to Google Cloud Console
2. Click on project dropdown at the top
3. Copy your Project ID

**In GitHub:**
1. **Settings** > **Secrets and variables** > **Actions** > **New repository secret**
2. Name: `GCP_PROJECT_ID`
3. Value: Your project ID (e.g., `my-project-12345`)

---

## Testing the CI/CD Pipeline

Once secrets are configured:

1. Push a commit to the `main` branch
2. Go to **Actions** tab in GitHub
3. Watch the workflow run:
   - ✅ Run Tests
   - ✅ Deploy to Google Cloud Run
   - ✅ Health Check

---

## Manual Deployment

If you need to deploy manually:

```bash
# Authenticate
gcloud auth login

# Deploy
gcloud run deploy malware-detector-py \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi
```

---

## Troubleshooting

### "Permission denied" errors
- Verify service account has correct IAM roles
- Check `GCP_SA_KEY` secret is valid JSON

### "Build failed" errors
- Check Dockerfile syntax
- Verify all dependencies in requirements-production.txt
- Review Cloud Build logs: `gcloud builds list`

### Health check fails
- Model files must be committed to git
- Check logs: `gcloud run services logs read malware-detector-py --region us-central1`
- Visit `/debug` endpoint for diagnostic info
