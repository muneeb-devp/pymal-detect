"""
Data preprocessing and feature engineering module
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class MalwarePreprocessor:
    """Handles data loading, cleaning, and preprocessing for malware detection"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'Label'
        
    def load_data(self, filepath):
        """Load the malware dataset"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Drop 'Identify' column as it has many missing values and is not useful
        if 'Identify' in df.columns:
            df = df.drop('Identify', axis=1)
        
        # Fill any remaining missing numeric values with median
        # Use pandas 3.0 compatible approach (no chained assignment)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df.loc[:, col] = df[col].fillna(df[col].median())
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        # Drop non-informative columns
        columns_to_drop = ['SHA1', 'FirstSeenDate', 'ImportedDlls', 'ImportedSymbols']
        df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
        
        return df
    
    def prepare_features_target(self, df):
        """Separate features and target variable"""
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        
        if self.feature_columns is None:
            self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into train and test sets with stratification"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Train class distribution:\n{y_train.value_counts(normalize=True)}")
        print(f"Test class distribution:\n{y_test.value_counts(normalize=True)}")
        
        return X_train, X_test, y_train, y_test
    
    def fit_scaler(self, X_train):
        """Fit the scaler on training data"""
        self.scaler.fit(X_train)
        return self
    
    def transform_features(self, X):
        """Transform features using the fitted scaler"""
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def fit_transform(self, X):
        """Fit and transform features"""
        self.fit_scaler(X)
        return self.transform_features(X)
    
    def preprocess_pipeline(self, filepath, test_size=0.2):
        """Complete preprocessing pipeline"""
        # Load data
        df = self.load_data(filepath)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
        
        # Fit scaler on training data only
        X_train_scaled = self.fit_transform(X_train)
        
        # Transform test data
        X_test_scaled = self.transform_features(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save the preprocessor for later use"""
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath='models/preprocessor.pkl'):
        """Load a saved preprocessor"""
        data = joblib.load(filepath)
        preprocessor = cls()
        preprocessor.scaler = data['scaler']
        preprocessor.feature_columns = data['feature_columns']
        return preprocessor
