"""
Evaluation script for trained model
Load test set and evaluate the production model
"""
import sys
import os
import json
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, 
    classification_report
)
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import MalwarePreprocessor


def main():
    print("="*80)
    print("MODEL EVALUATION ON TEST SET")
    print("="*80)
    
    # Load test set
    print("\nLoading test set...")
    test_df = pd.read_csv('data/test_set.csv')
    print(f"Test set size: {len(test_df)}")
    
    # Separate features and target
    y_test = test_df['Label']
    X_test = test_df.drop('Label', axis=1)
    
    # Load preprocessor
    print("\nLoading preprocessor...")
    preprocessor = MalwarePreprocessor.load_preprocessor('models/preprocessor.pkl')
    
    # Transform features
    print("Transforming features...")
    X_test_scaled = preprocessor.transform_features(X_test)
    
    # Load model
    print("\nLoading production model...")
    model = joblib.load('models/production_model.pkl')
    
    # Load metadata
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"\nModel: {metadata['model_name']}")
    print(f"CV AUC: {metadata['cv_auc']:.4f}")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nAUC:       {auc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               Goodware  Malware")
    print(f"Actual Goodware   {conf_matrix[0,0]:5d}    {conf_matrix[0,1]:5d}")
    print(f"       Malware    {conf_matrix[1,0]:5d}    {conf_matrix[1,1]:5d}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Goodware', 'Malware']))
    
    # Save evaluation results
    eval_results = {
        'model_name': metadata['model_name'],
        'test_auc': float(auc),
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'confusion_matrix': conf_matrix.tolist(),
        'test_size': len(y_test)
    }
    
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print("\nEvaluation results saved to results/evaluation_results.json")
    print("="*80)


if __name__ == '__main__':
    main()
