"""
Tests unitaires pour l'API FastAPI.
"""

from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch, MagicMock
from src.api import app
import numpy as np

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API de prédiction"}

def test_predict_valid_input():
    # Mock du PredictionService
    with patch('src.predict.PredictionService') as MockPredictService:
        # Configure le mock pour retourner une prédiction
        instance = MockPredictService.return_value
        instance.predict.return_value = np.array([1])

        # Teste l'API avec une entrée valide
        response = client.post(
            "/predict",
            json={"text": "exemple de texte"}
        )
        
        assert response.status_code == 200
        assert response.json() == {"prediction": 1.0}
        instance.predict.assert_called_once()

def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response.status_code == 400

def test_predict_invalid_json():
    response = client.post(
        "/predict",
        json={"invalid_field": "texte"}
    )
    assert response.status_code == 422  # Validation error

@pytest.mark.parametrize("error_type,expected_status", [
    (ValueError, 400),
    (Exception, 400)
])
def test_predict_error_handling(error_type, expected_status):
    with patch('src.predict.PredictionService') as MockPredictService:
        # Configure le mock pour lever une exception
        instance = MockPredictService.return_value
        instance.predict.side_effect = error_type("Erreur test")

        response = client.post(
            "/predict",
            json={"text": "texte problématique"}
        )
        
        assert response.status_code == expected_status
