"""
Tests unitaires pour le module de prédiction (src/predict.py).

Ce module teste directement le PredictionService, sans dépendance à l'API.
Les tests couvrent l'initialisation, les différentes façons de charger les modèles,
et les prédictions avec différents formats d'entrée.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from unittest.mock import mock_open, patch, MagicMock
import tensorflow


from src.predict import PredictionService
import pickle

@pytest.fixture
def mock_model():
    """Fixture qui crée un modèle simulé avec les méthodes predict et predict_proba (Mock : évite d'utiliser un vrai modèle entraîné)"""
    class MockModel:
        def predict(self, X):
            """Simule une prédiction binaire"""
            return np.array([1])  # 1 pour positif, 0 pour négatif
        
        def predict_proba(self, X):
            """Simule des probabilités de prédiction [négatif, positif]"""
            return np.array([[0.2, 0.8]])
    return MockModel()

@pytest.fixture
def mock_vectorizer():
    """Fixture qui crée un vectoriseur simulé avec la méthode transform (Mock : évite d'utiliser un vrai vectorizer entraîné)"""
    class MockVectorizer:
        def transform(self, texts):
            """Simule la transformation TF-IDF"""
            return np.array([[0.1, 0.2, 0.3]])
    return MockVectorizer()

def test_prediction_service_initialization():
    """Vérifie que le service de prédiction s'initialise correctement avec les chemins des modèles"""
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
def test_predict_with_different_inputs(input_text):
    """Vérifie que la méthode predict fonctionne avec différents formats d'entrée et les vrais modèles"""
    service = PredictionService(
        model_path="models/tf_idf_mdl.pkl",
        vectorizer_path="models/tf_idf_vectorizer.pkl"
    )
    prediction = service.predict(input_text)
    assert isinstance(prediction, np.ndarray)
    # Accepte (1,) ou (1, 2) selon la sortie du modèle
    assert prediction.shape in [(1,), (1, 2)]

@pytest.mark.parametrize("input_text", [
    "texte exemple",
    ["texte exemple"],
    pd.Series(["texte exemple"])
])
def test_predict_proba_with_different_inputs(input_text):
    """Vérifie que la méthode predict_proba fonctionne avec différents formats d'entrée et les vrais modèles"""
    service = PredictionService(
        model_path="models/tf_idf_mdl.pkl",
        vectorizer_path="models/tf_idf_vectorizer.pkl"
    )
    # Prépare l'entrée
    X = service._prepare_input(input_text)
    # Utilise predict_proba si disponible, sinon predict
    if hasattr(service.model, "predict_proba"):
        prediction = service.model.predict_proba(X)
    else:
        prediction = service.model.predict(X)
    assert isinstance(prediction, np.ndarray)
    # Si la sortie est 2D, on vérifie la somme à 1 (probabilités)
    if prediction.ndim == 2 and prediction.shape[1] == 2:
        assert np.isclose(np.sum(prediction[0]), 1.0)
    # Sinon, on vérifie juste la forme
    else:
        assert prediction.shape[0] == 1

def test_from_artifacts_constructor():
    """Vérifie la méthode factory from_artifacts qui crée un service avec des objets déjà chargés"""
    mock_model = MagicMock()
    mock_vectorizer = MagicMock()
    
    service = PredictionService.from_artifacts(
        model=mock_model,
        vectorizer=mock_vectorizer
    )
    
    # Vérifie que les objets sont correctement assignés
    assert service.model is mock_model
    assert service.vectorizer is mock_vectorizer
    # Vérifie que les attributs de chemin de fichier ne sont pas définis
    assert not hasattr(service, 'model_path')
    assert not hasattr(service, 'vectorizer_path')

def test_model_file_not_found():
    """Vérifie qu'une exception appropriée est levée lorsque le fichier du modèle n'existe pas"""
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

def test_prepare_input_formats():
    """Test que _prepare_input gère correctement différents formats d'entrée avec les vrais modèles"""
    service = PredictionService(
        model_path="models/tf_idf_mdl.pkl",
        vectorizer_path="models/tf_idf_vectorizer.pkl"
    )
    # Test avec string
    X = service._prepare_input("texte exemple")
    assert isinstance(X, np.ndarray)
    # Test avec liste
    X = service._prepare_input(["texte exemple"])
    assert isinstance(X, np.ndarray)
    # Test avec pandas.Series
    X = service._prepare_input(pd.Series(["texte exemple"]))
    assert isinstance(X, np.ndarray)
    # Test avec string
    service._prepare_input("texte exemple")
    mock_vectorizer.transform.assert_called_with(["texte exemple"])
    
    # Test avec liste
    service._prepare_input(["texte exemple"])
    mock_vectorizer.transform.assert_called_with(["texte exemple"])
    
    # Test avec pandas.Series
    service._prepare_input(pd.Series(["texte exemple"]))
    mock_vectorizer.transform.assert_called_with(["texte exemple"])
    
    # Test avec string
    service._prepare_input("texte exemple")
    mock_vectorizer.transform.assert_called_with(["texte exemple"])
    
    # Test avec liste
    service._prepare_input(["texte exemple"])
    mock_vectorizer.transform.assert_called_with(["texte exemple"])
    
    # Test avec pandas.Series
    service._prepare_input(pd.Series(["texte exemple"]))
    mock_vectorizer.transform.assert_called_with(["texte exemple"])
    