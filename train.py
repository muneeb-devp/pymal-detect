"""
Main training script for malware detection models
Run this script to:
1. Load and preprocess data
2. Train all baseline and advanced models with 10-fold CV
3. Select the best model
4. Evaluate on hold-out test set
5. Save the best model for production
"""
import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import MalwarePreprocessor
from models import ModelTrainer

# Set random seed for reproducibility
RANDOM_STATE = 42


def main():
    parser = argparse.ArgumentParser(description='Train malware detection models')
    parser.add_argument(
        '--data', 
        type=str, 
        default='brazilian-malware-dataset-master/brazilian-malware.csv',
        help='Path to the dataset CSV file'
    )
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Test set size (default: 0.2)'
    )
    parser.add_argument(
        '--cv-folds', 
        type=int, 
        default=10,
        help='Number of cross-validation folds (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Create directories for outputs
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    
    print("="*80)
    print("MALWARE DETECTION MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"Random State: {RANDOM_STATE}")
    print(f"Test Size: {args.test_size}")
    print(f"CV Folds: {args.cv_folds}")
    print("="*80)
    
    # Step 1: Data Preprocessing
    print("\n" + "="*80)
    print("STEP 1: DATA PREPROCESSING")
    print("="*80)
    
    preprocessor = MalwarePreprocessor(random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        args.data, 
        test_size=args.test_size
    )
    
    # Save preprocessor
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    # Save test set for later use
    import pandas as pd
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('data/test_set.csv', index=False)
    print(f"\nTest set saved to data/test_set.csv")
    
    # Step 2: Model Training with Cross-Validation
    print("\n" + "="*80)
    print("STEP 2: MODEL TRAINING AND CROSS-VALIDATION")
    print("="*80)
    
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    all_models = trainer.train_all_models(X_train, y_train)
    
    # Print summary
    trainer.print_summary()
    
    # Save CV results
    trainer.save_results('results/cv_results.json')
    
    # Step 3: Select Best Model
    print("\n" + "="*80)
    print("STEP 3: BEST MODEL SELECTION")
    print("="*80)
    
    best_model_name, best_model_results = trainer.get_best_model()
    
    # Step 4: Train Final Model and Evaluate on Test Set
    print("\n" + "="*80)
    print("STEP 4: FINAL MODEL TRAINING AND TEST SET EVALUATION")
    print("="*80)
    
    # Get fresh instance of best model
    input_dim = X_train.shape[1]
    if best_model_name == 'Logistic Regression':
        from sklearn.linear_model import LogisticRegression
        final_model = LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, n_jobs=1
        )
    elif best_model_name == 'Decision Tree':
        from sklearn.tree import DecisionTreeClassifier
        final_model = DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=10
        )
    elif best_model_name == 'Random Forest':
        from sklearn.ensemble import RandomForestClassifier
        final_model = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, 
            n_jobs=1, max_depth=10
        )
    elif best_model_name == 'PyTorch MLP':
        from models import MLPClassifier
        final_model = MLPClassifier(
            input_dim=input_dim, hidden_dims=[128, 64, 32],
            dropout=0.3, lr=0.001, epochs=20, batch_size=256,
            random_state=RANDOM_STATE
        )
    elif best_model_name == 'XGBoost':
        import xgboost as xgb
        final_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=1, eval_metric='logloss'
        )
    elif best_model_name == 'LightGBM':
        import lightgbm as lgb
        final_model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=1, verbosity=-1
        )
    elif best_model_name == 'CatBoost':
        from catboost import CatBoostClassifier
        final_model = CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=False
        )
    else:
        raise ValueError(f"Unknown model: {best_model_name}")
    
    # Train on full training set
    final_model = trainer.train_final_model(final_model, X_train, y_train)
    
    # Evaluate on test set
    test_results = trainer.evaluate_on_test(
        final_model, X_test, y_test, model_name=best_model_name
    )
    
    # Save test results
    import json
    with open('results/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=4)
    print("\nTest results saved to results/test_results.json")
    
    # Save final production model
    trainer.save_model(final_model, 'models/production_model.pkl')
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'cv_auc': best_model_results['cv_auc_mean'],
        'cv_auc_std': best_model_results['cv_auc_std'],
        'cv_accuracy': best_model_results['cv_accuracy_mean'],
        'cv_accuracy_std': best_model_results['cv_accuracy_std'],
        'test_auc': test_results['test_auc'],
        'test_accuracy': test_results['test_accuracy'],
        'feature_columns': preprocessor.feature_columns,
        'n_features': len(preprocessor.feature_columns),
        'random_state': RANDOM_STATE
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Model metadata saved to models/model_metadata.json")
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nBest Model: {best_model_name}")
    print(f"CV AUC: {best_model_results['cv_auc_mean']:.4f} ± {best_model_results['cv_auc_std']:.4f}")
    print(f"Test AUC: {test_results['test_auc']:.4f}")
    print(f"Test Accuracy: {test_results['test_accuracy']:.4f}")
    print("\nProduction model saved to: models/production_model.pkl")
    print("Preprocessor saved to: models/preprocessor.pkl")
    print("="*80)


if __name__ == '__main__':
    main()
