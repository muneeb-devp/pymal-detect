# 🛡️ Malware Detection System

A comprehensive machine learning-based static malware detection system that classifies Windows PE executables as malware or goodware. This project includes model training, web application deployment, automated testing, and CI/CD pipeline.

## � Live Demo

**Production URL:** [https://malware-detector-py-ypso2qakrq-uc.a.run.app/](https://malware-detector-py-ypso2qakrq-uc.a.run.app/)

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
├── models/                    # Saved models (generated after training)
├── results/                   # Training results (generated after training)
├── data/                      # Test set (generated after training)
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # CI/CD pipeline configuration
├── app.py                     # Flask web application
├── train.py                   # Main training script
├── requirements.txt           # Python dependencies
├── Procfile                   # Deployment configuration
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
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
    - Sets up Python environment
    - Installs dependencies
    - Runs all tests
    - Validates code imports

2. **On Push to Main (after tests pass):**
    - Triggers deployment to cloud platform
    - Performs health check

### Setup CI/CD

1. Add GitHub Secrets:
    - `RENDER_DEPLOY_HOOK` - Webhook URL from Render
    - `RENDER_APP_URL` - Deployed application URL

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

### Deploy to Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Configure build settings:
    - **Build Command**: `pip install -r requirements.txt`
    - **Start Command**: `gunicorn app:app`
4. Add environment variables if needed
5. Deploy!

### Deploy to Other Platforms

The app is compatible with:

- Heroku
- Railway
- Google Cloud Run
- AWS Elastic Beanstalk

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

This project was developed with assistance from AI code generation tools including:

- GitHub Copilot for code completion and suggestions
- AI-assisted debugging and optimization
- Documentation generation assistance

The AI tools significantly accelerated development while maintaining code quality through manual review and testing.

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

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: Model files are not included in the repository due to size. Run `python train.py` to generate them locally.
