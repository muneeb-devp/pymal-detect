"""
Simplified training script that avoids segfault issues
Trains only the most reliable models
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import catboost
from catboost import CatBoostClassifier
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import MalwarePreprocessor

RANDOM_STATE = 42

def main():
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    
    print("="*80)
    print("MALWARE DETECTION MODEL TRAINING - SIMPLIFIED")
    print("="*80)
    
    # Step 1: Preprocessing
    print("\nStep 1: Data Preprocessing")
    print("-"*80)
    
    preprocessor = MalwarePreprocessor(random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        'brazilian-malware-dataset-master/brazilian-malware.csv',
        test_size=0.2
    )
    
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    # Save test set
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('data/test_set.csv', index=False)
    print("Test set saved")
    
    # Step 2: Model Training
    print("\nStep 2: Model Training with 3-Fold CV")
    print("-"*80)
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, max_depth=10, n_jobs=1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=1, eval_metric='logloss'
        ),
        'CatBoost': CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=False
        )
    }
    
    results = {}
    cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        auc_scores = cross_val_score(
            model, X_train, y_train, cv=cv_splitter,
            scoring='roc_auc', n_jobs=1
        )
        acc_scores = cross_val_score(
            model, X_train, y_train, cv=cv_splitter,
            scoring='accuracy', n_jobs=1
        )
        
        results[name] = {
            'cv_auc_mean': float(auc_scores.mean()),
            'cv_auc_std': float(auc_scores.std()),
            'cv_accuracy_mean': float(acc_scores.mean()),
            'cv_accuracy_std': float(acc_scores.std())
        }
        
        print(f"  CV AUC: {results[name]['cv_auc_mean']:.4f} ± {results[name]['cv_auc_std']:.4f}")
        print(f"  CV Acc: {results[name]['cv_accuracy_mean']:.4f} ± {results[name]['cv_accuracy_std']:.4f}")
    
    # Save CV results
    with open('results/cv_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Step 3: Select best model
    print("\nStep 3: Selecting Best Model")
    print("-"*80)
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_auc_mean'])
    print(f"\nBest Model: {best_model_name}")
    print(f"CV AUC: {results[best_model_name]['cv_auc_mean']:.4f}")
    
    # Step 4: Train final model and evaluate
    print("\nStep 4: Training Final Model on Full Training Set")
    print("-"*80)
    
    final_model = models[best_model_name]
    print(f"Training {best_model_name}...")
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\nEvaluating on Test Set...")
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nTest AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save results
    test_results = {
        'model_name': best_model_name,
        'test_auc': float(test_auc),
        'test_accuracy': float(test_accuracy),
        'confusion_matrix': conf_matrix.tolist()
    }
    
    with open('results/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=4)
    
    # Save model
    joblib.dump(final_model, 'models/production_model.pkl')
    print("\nModel saved to models/production_model.pkl")
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'cv_auc': results[best_model_name]['cv_auc_mean'],
        'cv_auc_std': results[best_model_name]['cv_auc_std'],
        'cv_accuracy': results[best_model_name]['cv_accuracy_mean'],
        'cv_accuracy_std': results[best_model_name]['cv_accuracy_std'],
        'test_auc': float(test_auc),
        'test_accuracy': float(test_accuracy),
        'feature_columns': preprocessor.feature_columns,
        'n_features': len(preprocessor.feature_columns),
        'random_state': RANDOM_STATE
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("="*80)

if __name__ == '__main__':
    main()
