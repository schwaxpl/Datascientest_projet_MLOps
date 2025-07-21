"""
Tests unitaires pour le service de prédiction.
"""

import pytest
import pandas as pd
import numpy as np
from src.predict import PredictionService
import pickle
from unittest.mock import mock_open, patch

@pytest.fixture
def mock_model():
    class MockModel:
        def predict(self, X):
            # Simule une prédiction binaire
            return np.array([1])
    return MockModel()

@pytest.fixture
def mock_vectorizer():
    class MockVectorizer:
        def transform(self, texts):
            # Simule la transformation TF-IDF
            return np.array([[0.1, 0.2, 0.3]])
    return MockVectorizer()

def test_prediction_service_initialization():
    service = PredictionService(
        model_path="models/tf_idf_mdl.pkl",
        vectorizer_path="models/tf_idf_vectorizer.pkl"
    )
    assert service.model_path == "models/tf_idf_mdl.pkl"
    assert service.vectorizer_path == "models/tf_idf_vectorizer.pkl"

@pytest.mark.parametrize("input_text", [
    "texte exemple",
    ["texte exemple"],
    pd.Series(["texte exemple"])
])
def test_predict_with_different_inputs(input_text, mock_model, mock_vectorizer):
    with patch('builtins.open', mock_open()):
        with patch('pickle.load') as mock_pickle:
            # Configure mock pour retourner le modèle puis le vectorizer
            mock_pickle.side_effect = [mock_model, mock_vectorizer]
            
            service = PredictionService(
                model_path="models/tf_idf_mdl.pkl",
                vectorizer_path="models/tf_idf_vectorizer.pkl"
            )
            
            prediction = service.predict(input_text)
            assert isinstance(prediction, np.ndarray)
            assert prediction.shape == (1,)

def test_model_file_not_found():
    with pytest.raises(FileNotFoundError):
        PredictionService(
            model_path="nonexistent/model.pkl",
            vectorizer_path="models/tf_idf_vectorizer.pkl"
        )

def test_vectorizer_file_not_found():
    with pytest.raises(FileNotFoundError):
        PredictionService(
            model_path="models/tf_idf_mdl.pkl",
            vectorizer_path="nonexistent/vectorizer.pkl"
        )
