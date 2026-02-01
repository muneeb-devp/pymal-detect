"""
Flask Web Application for Malware Detection
Features:
- Manual feature entry for single predictions
- Batch file upload for multiple predictions
- Evaluation metrics display when labels are provided
"""
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import io

app = Flask(__name__)
CORS(app)

# Load model and preprocessor
MODEL_PATH = 'models/production_model.pkl'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'
METADATA_PATH = 'models/model_metadata.json'

# Global variables for model
model = None
preprocessor = None
metadata = None
feature_columns = None


def load_model_artifacts():
    """Load model, preprocessor, and metadata"""
    global model, preprocessor, metadata, feature_columns
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    
    try:
        from src.preprocessing import MalwarePreprocessor
        preprocessor = MalwarePreprocessor.load_preprocessor(PREPROCESSOR_PATH)
        feature_columns = preprocessor.feature_columns
        print(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
    except Exception as e:
        print(f"Error loading preprocessor: {e}")
        preprocessor = None
    
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata loaded from {METADATA_PATH}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        metadata = None


# Load artifacts at startup
load_model_artifacts()


@app.route('/')
def index():
    """Home page with input form"""
    return render_template('index.html', 
                          feature_columns=feature_columns,
                          metadata=metadata)


@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'metadata_loaded': metadata is not None
    }
    return jsonify(status), 200 if status['status'] == 'healthy' else 503


@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    try:
        if model is None or preprocessor is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get features from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Create DataFrame with feature values
        features_dict = {}
        for col in feature_columns:
            if col in data:
                features_dict[col] = [float(data[col])]
            else:
                return jsonify({'error': f'Missing feature: {col}'}), 400
        
        df = pd.DataFrame(features_dict)
        
        # Transform features
        df_scaled = preprocessor.transform_features(df)
        
        # Make prediction
        prediction = model.predict(df_scaled)[0]
        prediction_proba = model.predict_proba(df_scaled)[0]
        
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Malware' if prediction == 1 else 'Goodware',
            'probability': {
                'goodware': float(prediction_proba[0]),
                'malware': float(prediction_proba[1])
            },
            'confidence': float(max(prediction_proba))
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for uploaded CSV file"""
    try:
        if model is None or preprocessor is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Check if Label column exists (for evaluation)
        has_labels = 'Label' in df.columns
        
        if has_labels:
            y_true = df['Label'].values
            df_features = df.drop('Label', axis=1)
        else:
            y_true = None
            df_features = df
        
        # Check if all required features are present
        missing_features = set(feature_columns) - set(df_features.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing features: {list(missing_features)}'
            }), 400
        
        # Select only required features in correct order
        df_features = df_features[feature_columns]
        
        # Transform features
        df_scaled = preprocessor.transform_features(df_features)
        
        # Make predictions
        predictions = model.predict(df_scaled)
        predictions_proba = model.predict_proba(df_scaled)
        
        result = {
            'num_samples': len(predictions),
            'predictions': predictions.tolist(),
            'probabilities': predictions_proba.tolist(),
            'summary': {
                'num_malware': int(np.sum(predictions == 1)),
                'num_goodware': int(np.sum(predictions == 0)),
                'percent_malware': float(np.mean(predictions == 1) * 100)
            }
        }
        
        # If labels are provided, calculate evaluation metrics
        if has_labels:
            auc = roc_auc_score(y_true, predictions_proba[:, 1])
            accuracy = accuracy_score(y_true, predictions)
            conf_matrix = confusion_matrix(y_true, predictions)
            
            result['evaluation'] = {
                'auc': float(auc),
                'accuracy': float(accuracy),
                'confusion_matrix': conf_matrix.tolist(),
                'true_positives': int(conf_matrix[1, 1]),
                'true_negatives': int(conf_matrix[0, 0]),
                'false_positives': int(conf_matrix[0, 1]),
                'false_negatives': int(conf_matrix[1, 0])
            }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info')
def model_info():
    """Get model information"""
    if metadata is None:
        return jsonify({'error': 'Metadata not loaded'}), 500
    
    return jsonify(metadata), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8081))
    app.run(host='0.0.0.0', port=port, debug=False)
