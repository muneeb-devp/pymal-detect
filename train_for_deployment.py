#!/usr/bin/env python3
"""
Quick training script for deployment - trains only the best model (LightGBM)
This is optimized for fast deployment on platforms like Render.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import MalwarePreprocessor
import lightgbm as lgb
import joblib
import json
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import datetime

RANDOM_STATE = 42
DATA_PATH = 'brazilian-malware-dataset-master/brazilian-malware.csv'


def train_deployment_model():
    """Train and save the production model quickly"""
    print("="*80)
    print("DEPLOYMENT MODEL TRAINING")
    print("="*80)
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    
    # Check if model already exists
    if os.path.exists('models/production_model.pkl'):
        print("Model already exists. Skipping training.")
        return
    
    print("\n1. Loading and preprocessing data...")
    
    # Load data
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df)} samples")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}")
        print("Please ensure the dataset is available for training.")
        sys.exit(1)
    
    # Preprocess
    preprocessor = MalwarePreprocessor(random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        DATA_PATH, 
        test_size=0.2
    )
    
    # Save preprocessor
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    print("Preprocessor saved")
    
    # Save test set
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('data/test_set.csv', index=False)
    print("Test set saved")
    
    print("\n2. Training LightGBM model...")
    
    # Train LightGBM (best performing model)
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training complete")
    
    print("\n3. Evaluating model...")
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\n4. Saving model and metadata...")
    
    # Save model
    joblib.dump(model, 'models/production_model.pkl')
    print("Model saved to models/production_model.pkl")
    
    # Save metadata
    metadata = {
        'model_name': 'LightGBM',
        'model_type': 'LGBMClassifier',
        'test_auc': float(test_auc),
        'test_accuracy': float(test_accuracy),
        'n_features': X_train.shape[1],
        'feature_names': list(X_train.columns),
        'training_date': datetime.now().isoformat(),
        'random_state': RANDOM_STATE,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved")
    
    print("\n" + "="*80)
    print("DEPLOYMENT MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Model: LightGBM")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("="*80)


if __name__ == '__main__':
    train_deployment_model()
