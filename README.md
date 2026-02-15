# 🛡️ Malware Detection System

A comprehensive machine learning-based static malware detection system that classifies Windows PE executables as malware or goodware. This project includes model training, web application deployment, automated testing, and CI/CD pipeline.

## � Live Demo

**Production URL:** [https://malware-detector-py-ypso2qakrq-uc.a.run.app/](https://malware-detector-py-ypso2qakrq-uc.a.run.app/)
**Deployed on:** Google Cloud Run (us-central1)
Try the live application to analyze PE files for malware detection!

## �🎯 Project Overview

This system performs **static malware detection** by analyzing features extracted from Windows Portable Executable (PE) files without executing them. The project demonstrates the complete ML pipeline from data preprocessing to production deployment.

### Key Features

- **Multiple ML Models**: Logistic Regression, Decision Tree, Random Forest, PyTorch MLP, XGBoost, LightGBM, and CatBoost
- **Robust Evaluation**: 10-fold stratified cross-validation with 80/20 train/test split
- **Web Application**: Flask-based UI for single predictions and batch processing
- **Automated Testing**: Comprehensive unit and integration tests
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Production Ready**: Deployed on cloud platform with health monitoring

## 📊 Dataset

The project uses the Brazilian Malware Dataset containing ~50,000 PE file samples with 27 features:

- **Samples**: 50,181 instances
- **Features**: 22 numeric features (after preprocessing)
- **Classes**: Malware (1) and Goodware (0)
- **Class Distribution**: ~58% Malware, ~42% Goodware

## 🏗️ Project Structure

```
pyMalDetect/
├── src/
│   ├── preprocessing.py      # Data preprocessing and feature engineering
│   └── models.py              # Model training and evaluation
├── tests/
│   ├── test_preprocessing.py # Unit tests for preprocessing
│   └── test_api.py            # Integration tests for API
├── templates/
│   └── index.html             # Web interface
├── models/                    # Saved models (334KB LightGBM)
│   ├── production_model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.json
├── results/                   # Training results
│   ├── cv_results.json
│   └── test_results.json
├── data/                      # Test set
│   └── test_set.csv
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # CI/CD pipeline (Google Cloud Run)
├── app.py                     # Flask web application
├── train.py                   # Main training script
├── train_for_deployment.py    # Training script for production
├── requirements.txt           # Development dependencies
├── requirements-production.txt # Production dependencies (minimal)
├── Dockerfile                 # Container configuration
├── deployed.md                # Deployment details and URL
├── evaluation-and-design.md   # Model evaluation and design decisions
├── ai-tooling.md              # AI tools usage report
├── CI-CD-SETUP.md             # CI/CD configuration guide
├── SUBMISSION-CHECKLIST.md    # Project submission checklist
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip
- Virtual environment (recommended)
- Git

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd pyMalDetect
```

2. **Create and activate virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Training Models

Train all models with 10-fold cross-validation:

```bash
python train.py --cv-folds 10
```

This will:

- Load and preprocess the data
- Split into 80% training and 20% test sets
- Train 7 different models with cross-validation
- Select the best model
- Evaluate on hold-out test set
- Save the production model and metadata

**Expected output:**

- `models/production_model.pkl` - Best performing model
- `models/preprocessor.pkl` - Data preprocessor
- `models/model_metadata.json` - Model information
- `results/cv_results.json` - Cross-validation results
- `results/test_results.json` - Final test set results
- `data/test_set.csv` - Hold-out test set

### Running the Web Application

Start the Flask server:

```bash
python app.py
```

Access the web interface at `http://localhost:5000`

## 🌐 Web Application Features

### 1. Single Prediction

- Manual entry of PE file features
- "Load Demo Data" button for quick testing
- Real-time malware classification
- Confidence scores and probabilities

### 2. Batch Upload

- Upload CSV files with multiple samples
- Bulk predictions
- Summary statistics (malware %, counts)
- Evaluation metrics when labels are provided:
    - AUC (Area Under ROC Curve)
    - Accuracy
    - Confusion Matrix

### API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions
- `GET /model_info` - Model metadata

## 🧪 Testing

Run all tests:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
# Unit tests
pytest tests/test_preprocessing.py -v

# Integration tests
pytest tests/test_api.py -v
```

Generate coverage report:

```bash
pytest tests/ --cov=src --cov-report=html
```

## 🔄 CI/CD Pipeline

The project includes a GitHub Actions workflow that:

1. **On Pull Request or Push to Main:**
    - Sets up Python 3.13 environment
    - Installs dependencies
    - Runs all tests
    - Validates code imports

2. **On Push to Main (after tests pass):**
    - Authenticates with Google Cloud
    - Builds Docker container using Cloud Build
    - Deploys to Google Cloud Run
    - Performs automated health check

### Setup CI/CD

1. Add GitHub Secrets:
    - `GCP_SA_KEY` - Google Cloud Service Account JSON key
    - `GCP_PROJECT_ID` - Your Google Cloud Project ID

2. See [CI-CD-SETUP.md](CI-CD-SETUP.md) for detailed configuration instructions

## 📈 Model Performance

### Best Model Results

Based on cross-validation, the models typically achieve:

- **AUC**: > 0.99
- **Accuracy**: > 0.97

The Random Forest or gradient boosting models (XGBoost, LightGBM, CatBoost) usually perform best on this dataset.

### Model Comparison

All models are evaluated using:

- **Primary Metric**: AUC (Area Under ROC Curve)
- **Secondary Metric**: Accuracy
- **Validation**: 10-fold stratified cross-validation
- **Final Evaluation**: Hold-out test set (20% of data)

## 🚢 Deployment

### Current Deployment: Google Cloud Run

The application is deployed using Docker containers on Google Cloud Run:

**Features:**
- Automatic scaling (0 to multiple instances)
- Pay-per-use pricing
- HTTPS by default
- Global CDN
- 1GB RAM, 300s timeout

**Deployment Process:**

1. Code pushed to `main` branch triggers GitHub Actions
2. Docker image built using Cloud Build
3. Image pushed to Artifact Registry
4. Service deployed to Cloud Run
5. Automated health check performed

### Manual Deployment

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

### Deploy to Other Platforms

The app is compatible with:

- **Google Cloud Run** ✅ (Current)
- Heroku
- Railway
- AWS App Runner
- Azure Container Apps

## 🛠️ Development

### Adding New Models

1. Add model class in `src/models.py`
2. Update `get_advanced_models()` or `get_baseline_models()`
3. Retrain with `python train.py`

### Modifying Preprocessing

1. Update `src/preprocessing.py`
2. Retrain models to regenerate preprocessor

### Adding Features

1. Update preprocessing pipeline
2. Ensure new features are in training data
3. Retrain models
4. Update web form in `templates/index.html`

## 📝 AI Tool Usage

This project was developed with assistance from AI code generation tools:

- **GitHub Copilot**: Code completion and suggestions
- **Claude (Anthropic)**: Architecture decisions, debugging, deployment troubleshooting
- **AI-assisted tasks**: Documentation, test generation, CI/CD configuration

The AI tools provided an estimated **5x productivity boost** while maintaining code quality through manual review and testing. See [ai-tooling.md](ai-tooling.md) for detailed usage report.

## 🔬 Technical Details

### Data Preprocessing

- Missing value imputation (median for numeric features)
- Feature standardization (StandardScaler)
- Removal of non-informative features (SHA1, dates, etc.)
- Stratified train/test split to preserve class balance

### Cross-Validation Strategy

- 10-fold stratified cross-validation
- Separate preprocessing fit on each fold
- Prevents data leakage
- Fixed random seed for reproducibility

### Model Training

- Scikit-learn for traditional ML models
- PyTorch for neural network
- Gradient boosting frameworks (XGBoost, LightGBM, CatBoost)
- Parallel processing where supported

## 📄 License

This project is for educational purposes.

## 📚 Additional Documentation

- **[deployed.md](deployed.md)** - Production deployment details and API endpoints
- **[evaluation-and-design.md](evaluation-and-design.md)** - Comprehensive model evaluation with cross-validation results, test metrics, and design decisions
- **[ai-tooling.md](ai-tooling.md)** - Detailed report on AI tools usage, productivity impact, and best practices
- **[CI-CD-SETUP.md](CI-CD-SETUP.md)** - Step-by-step guide for configuring GitHub Actions with Google Cloud
- **[SUBMISSION-CHECKLIST.md](SUBMISSION-CHECKLIST.md)** - Project submission requirements and status

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: Model files (334KB) are included in the repository for deployment convenience. Run `python train.py` to retrain models if needed.
