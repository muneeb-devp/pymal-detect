"""
Model training and evaluation module
Includes baseline models (Logistic Regression, Decision Tree, Random Forest, PyTorch MLP)
and advanced models (XGBoost, LightGBM, CatBoost)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, 
    classification_report, make_scorer
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json
from datetime import datetime


class PyTorchMLP(nn.Module):
    """Multi-layer Perceptron for malware classification"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super(PyTorchMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPClassifier:
    """Wrapper for PyTorch MLP to match sklearn interface"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], 
                 dropout=0.3, lr=0.001, epochs=50, batch_size=256, 
                 random_state=42):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        # Use CPU to avoid MPS issues on Mac
        self.device = torch.device('cpu')
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'lr': self.lr,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def fit(self, X, y):
        """Train the MLP"""
        self.model = PyTorchMLP(
            self.input_dim, 
            self.hidden_dims, 
            self.dropout
        ).to(self.device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y).reshape(-1, 1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy()
        
        # Return probabilities for both classes
        proba_class_1 = outputs.flatten()
        proba_class_0 = 1 - proba_class_1
        return np.column_stack([proba_class_0, proba_class_1])
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class ModelTrainer:
    """Handles training and evaluation of multiple models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.trained_models = {}
    
    def get_baseline_models(self, input_dim):
        """Get baseline models"""
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=1  # Changed from -1
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=1,  # Changed from -1
                max_depth=10
            )
            # PyTorch MLP removed - causes segmentation faults with cross_validate on macOS
            # 'PyTorch MLP': MLPClassifier(
            #     input_dim=input_dim,
            #     hidden_dims=[128, 64, 32],
            #     dropout=0.3,
            #     lr=0.001,
            #     epochs=20,  # Reduced for faster training
            #     batch_size=256,
            #     random_state=self.random_state
            # )
        }
        return models
    
    def get_advanced_models(self):
        """Get advanced gradient boosting models"""
        models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=1,  # Changed from -1
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=1,  # Changed from -1
                verbosity=-1
            ),
            'CatBoost': CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=False
            )
        }
        return models
    
    def evaluate_model_cv(self, model, X, y, cv=10, model_name='Model'):
        """Evaluate model using cross-validation"""
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print('='*60)
        
        cv_splitter = StratifiedKFold(
            n_splits=cv, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        scoring = {
            'auc': 'roc_auc',
            'accuracy': 'accuracy'
        }
        
        # Disable parallel processing to avoid segfaults
        cv_results = cross_validate(
            model, X, y,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=1,  # Changed from -1 to 1
            return_train_score=True
        )
        
        results = {
            'model_name': model_name,
            'cv_auc_mean': cv_results['test_auc'].mean(),
            'cv_auc_std': cv_results['test_auc'].std(),
            'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
            'cv_accuracy_std': cv_results['test_accuracy'].std(),
            'train_auc_mean': cv_results['train_auc'].mean(),
            'train_accuracy_mean': cv_results['train_accuracy'].mean()
        }
        
        print(f"CV AUC: {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
        print(f"CV Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
        
        self.results[model_name] = results
        
        return results
    
    def train_all_models(self, X_train, y_train):
        """Train all baseline and advanced models with cross-validation"""
        input_dim = X_train.shape[1]
        
        print("\n" + "="*60)
        print("TRAINING BASELINE MODELS")
        print("="*60)
        
        baseline_models = self.get_baseline_models(input_dim)
        for name, model in baseline_models.items():
            self.evaluate_model_cv(model, X_train, y_train, cv=10, model_name=name)
        
        print("\n" + "="*60)
        print("TRAINING ADVANCED MODELS")
        print("="*60)
        
        advanced_models = self.get_advanced_models()
        for name, model in advanced_models.items():
            self.evaluate_model_cv(model, X_train, y_train, cv=10, model_name=name)
        
        # Combine all models
        all_models = {**baseline_models, **advanced_models}
        
        return all_models
    
    def get_best_model(self):
        """Get the best performing model based on CV AUC"""
        if not self.results:
            raise ValueError("No models have been evaluated yet")
        
        best_model_name = max(
            self.results.keys(), 
            key=lambda x: self.results[x]['cv_auc_mean']
        )
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"CV AUC: {self.results[best_model_name]['cv_auc_mean']:.4f}")
        print('='*60)
        
        return best_model_name, self.results[best_model_name]
    
    def train_final_model(self, model, X_train, y_train):
        """Train the final model on full training set"""
        print("\nTraining final model on full training set...")
        model.fit(X_train, y_train)
        return model
    
    def evaluate_on_test(self, model, X_test, y_test, model_name='Model'):
        """Evaluate model on hold-out test set"""
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION ON TEST SET: {model_name}")
        print('='*60)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"\nTest AUC: {auc:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        results = {
            'model_name': model_name,
            'test_auc': auc,
            'test_accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def save_results(self, filepath='results/model_results.json'):
        """Save training results"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\nResults saved to {filepath}")
    
    def save_model(self, model, filepath='models/best_model.pkl'):
        """Save trained model"""
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath='models/best_model.pkl'):
        """Load saved model"""
        return joblib.load(filepath)
    
    def print_summary(self):
        """Print summary of all model results"""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY (sorted by CV AUC)")
        print("="*60)
        
        # Sort by CV AUC
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['cv_auc_mean'],
            reverse=True
        )
        
        print(f"\n{'Model':<20} {'CV AUC':<15} {'CV Accuracy':<15}")
        print("-" * 60)
        
        for model_name, results in sorted_results:
            auc_str = f"{results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}"
            acc_str = f"{results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}"
            print(f"{model_name:<20} {auc_str:<15} {acc_str:<15}")
