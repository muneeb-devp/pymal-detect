# Deployed Application

## Production URL

**Live Application:** [https://malware-detector-py-ypso2qakrq-uc.a.run.app/](https://malware-detector-py-ypso2qakrq-uc.a.run.app/)

## Deployment Details

- **Platform:** Google Cloud Run
- **Region:** us-central1 (Iowa)
- **Service Name:** malware-detector-py
- **Runtime:** Python 3.11
- **Memory:** 1 GB
- **CPU:** 1 vCPU
- **Concurrency:** 80 requests per instance
- **Timeout:** 300 seconds

## Health Check

To verify the application is running:

```bash
curl https://malware-detector-py-ypso2qakrq-uc.a.run.app/health
```

**Expected Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "preprocessor_loaded": true,
    "metadata_loaded": true,
    "feature_count": 41,
    "errors": []
}
```

## Features

The deployed application provides:

1. **Single File Analysis** - Manual feature entry with demo data loader
2. **Batch File Analysis** - CSV file upload for multiple predictions
3. **Real-time Predictions** - Instant malware classification
4. **Model Information** - View trained model metrics and metadata

## API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check endpoint
- `GET /debug` - Debug information (for troubleshooting)
- `POST /predict` - Single prediction API
- `POST /predict_batch` - Batch prediction API
- `GET /model_info` - Model metadata

## CI/CD Pipeline

The application is automatically deployed via GitHub Actions on every push to the main branch. The CI/CD pipeline:

1. Runs unit tests
2. Runs integration tests
3. Verifies module imports
4. Deploys to Google Cloud Run
5. Performs health check

## Model Details

- **Algorithm:** LightGBM Gradient Boosting
- **Training Set:** 40,145 samples
- **Test Set:** 10,036 samples
- **Features:** 41 PE file characteristics
- **Test Accuracy:** 98.06%
- **Test AUC-ROC:** 0.9962

## Monitoring

View application logs:
```bash
gcloud run services logs read malware-detector-py --region us-central1
```

## Build Information

- **Base Image:** python:3.11-slim
- **System Dependencies:** libgomp1 (for LightGBM)
- **Python Dependencies:** See `requirements-production.txt`
- **Build Time:** ~3-4 minutes
- **Image Size:** ~150 MB (optimized)

## Security

- HTTPS enforced
- CORS enabled for cross-origin requests
- No authentication required (public demo)
- Input validation on all endpoints

## Last Updated

February 15, 2026
