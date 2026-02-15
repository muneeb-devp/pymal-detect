# Lightweight Docker image for malware detection app
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files
COPY requirements-production.txt .
COPY app.py .
COPY train_for_deployment.py .
COPY Procfile .
COPY setup.sh .
COPY src/ ./src/
COPY templates/ ./templates/
COPY models/ ./models/
COPY data/test_set.csv ./data/

# Install dependencies
RUN pip install --no-cache-dir -r requirements-production.txt

# Verify model files are present
RUN echo "Checking for model files..." && \
    ls -lah models/ && \
    test -f models/production_model.pkl || (echo "ERROR: production_model.pkl not found!" && exit 1) && \
    test -f models/preprocessor.pkl || (echo "ERROR: preprocessor.pkl not found!" && exit 1) && \
    test -f models/model_metadata.json || (echo "ERROR: model_metadata.json not found!" && exit 1) && \
    echo "✓ All model files present"

# Expose port
EXPOSE 8080

# Run the application
# Use PORT env variable for Cloud Run compatibility
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --timeout 120 --workers 2
