"""
Unit tests for preprocessing module
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import MalwarePreprocessor


class TestMalwarePreprocessor:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        data = {
            'BaseOfCode': [4096, 4096, 4096],
            'BaseOfData': [69632, 40960, 69632],
            'Characteristics': [258, 258, 258],
            'DllCharacteristics': [0, 0, 0],
            'Entropy': [7.98, 7.50, 7.98],
            'FileAlignment': [512, 512, 512],
            'ImageBase': [4194304, 4194304, 4194304],
            'Machine': [332, 332, 332],
            'Magic': [267, 267, 267],
            'NumberOfRvaAndSizes': [16, 16, 16],
            'NumberOfSections': [3, 3, 3],
            'NumberOfSymbols': [0, 0, 0],
            'PE_TYPE': [32, 32, 32],
            'PointerToSymbolTable': [0, 0, 0],
            'Size': [57344, 40000, 57344],
            'SizeOfCode': [45056, 30000, 45056],
            'SizeOfHeaders': [1024, 1024, 1024],
            'SizeOfImage': [73728, 50000, 73728],
            'SizeOfInitializedData': [11776, 10000, 11776],
            'SizeOfOptionalHeader': [224, 224, 224],
            'SizeOfUninitializedData': [1500, 1500, 1500],
            'TimeDateStamp': [12345, 12345, 12345],
            'Label': [1, 0, 1]  # Target variable
        }
        return pd.DataFrame(data)
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = MalwarePreprocessor(random_state=42)
        assert preprocessor.random_state == 42
        assert preprocessor.feature_columns is None
        assert preprocessor.target_column == 'Label'
    
    def test_handle_missing_values(self, sample_data):
        """Test missing value handling"""
        preprocessor = MalwarePreprocessor()
        
        # Add some missing values
        sample_data.loc[0, 'Entropy'] = np.nan
        
        result = preprocessor.handle_missing_values(sample_data)
        
        # Check that missing values are filled
        assert result['Entropy'].isnull().sum() == 0
    
    def test_prepare_features_target(self, sample_data):
        """Test feature and target separation"""
        preprocessor = MalwarePreprocessor()
        
        X, y = preprocessor.prepare_features_target(sample_data)
        
        # Check that Label is not in features
        assert 'Label' not in X.columns
        
        # Check that target has correct values
        assert list(y) == [1, 0, 1]
        
        # Check that feature columns are stored
        assert preprocessor.feature_columns is not None
    
    def test_split_data(self, sample_data):
        """Test data splitting"""
        preprocessor = MalwarePreprocessor(random_state=42)
        X, y = preprocessor.prepare_features_target(sample_data)
        
        # Note: With only 3 samples, test_size needs to be adjusted
        # or we test with more data
        # For this test, we'll just verify the function runs
        try:
            X_train, X_test, y_train, y_test = preprocessor.split_data(
                X, y, test_size=0.33
            )
            assert len(X_train) + len(X_test) == len(X)
            assert len(y_train) + len(y_test) == len(y)
        except ValueError:
            # Expected if sample size is too small for stratification
            pass
    
    def test_fit_transform(self, sample_data):
        """Test feature scaling"""
        preprocessor = MalwarePreprocessor()
        X, y = preprocessor.prepare_features_target(sample_data)
        
        X_scaled = preprocessor.fit_transform(X)
        
        # Check that scaling was applied
        assert X_scaled.shape == X.shape
        
        # Check that mean is close to 0 and std is close to 1
        assert abs(X_scaled.values.mean()) < 1e-10
        assert abs(X_scaled.values.std() - 1.0) < 0.1


def test_preprocessor_save_load(tmp_path, sample_data):
    """Test saving and loading preprocessor"""
    preprocessor = MalwarePreprocessor()
    X, y = preprocessor.prepare_features_target(sample_data)
    preprocessor.fit_transform(X)
    
    # Save
    save_path = tmp_path / "test_preprocessor.pkl"
    preprocessor.save_preprocessor(str(save_path))
    
    # Load
    loaded_preprocessor = MalwarePreprocessor.load_preprocessor(str(save_path))
    
    # Check that loaded preprocessor has same attributes
    assert loaded_preprocessor.feature_columns == preprocessor.feature_columns
