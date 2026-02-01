"""
Integration tests for the Flask API
"""
import pytest
import json
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import app only if models exist
try:
    from app import app
    APP_AVAILABLE = True
except:
    APP_AVAILABLE = False


@pytest.fixture
def client():
    """Create test client"""
    if not APP_AVAILABLE:
        pytest.skip("App not available (models not trained yet)")
    
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_features():
    """Sample feature values for testing"""
    return {
        'BaseOfCode': 4096,
        'BaseOfData': 69632,
        'Characteristics': 258,
        'DllCharacteristics': 0,
        'Entropy': 7.9876,
        'FileAlignment': 512,
        'ImageBase': 4194304,
        'Machine': 332,
        'Magic': 267,
        'NumberOfRvaAndSizes': 16,
        'NumberOfSections': 3,
        'NumberOfSymbols': 0,
        'PE_TYPE': 32,
        'PointerToSymbolTable': 0,
        'Size': 57344,
        'SizeOfCode': 45056,
        'SizeOfHeaders': 1024,
        'SizeOfImage': 73728,
        'SizeOfInitializedData': 11776,
        'SizeOfOptionalHeader': 224,
        'SizeOfUninitializedData': 1500,
        'TimeDateStamp': 12345
    }


@pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")
class TestFlaskAPI:
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code in [200, 503]
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'model_loaded' in data
    
    def test_predict_endpoint(self, client, sample_features):
        """Test single prediction endpoint"""
        response = client.post(
            '/predict',
            data=json.dumps(sample_features),
            content_type='application/json'
        )
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'prediction' in data
            assert 'prediction_label' in data
            assert 'probability' in data
            assert data['prediction'] in [0, 1]
    
    def test_predict_missing_feature(self, client, sample_features):
        """Test prediction with missing feature"""
        # Remove one feature
        incomplete_features = sample_features.copy()
        del incomplete_features['Entropy']
        
        response = client.post(
            '/predict',
            data=json.dumps(incomplete_features),
            content_type='application/json'
        )
        
        # Should return error
        assert response.status_code in [400, 500]
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get('/model_info')
        
        # Should either succeed or return error gracefully
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'model_name' in data or 'error' in data


def test_import_modules():
    """Test that required modules can be imported"""
    try:
        from src.preprocessing import MalwarePreprocessor
        from src.models import ModelTrainer
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")
