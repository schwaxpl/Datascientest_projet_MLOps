"""
Tests unitaires pour les utilitaires et la configuration du projet MLOps.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile
import pickle
import mlflow
from src.utils import load_model_from_registry, get_latest_registered_version
from src.config import REQUIRED_COLUMNS, INGESTION_EXPERIMENT_NAME

def test_required_columns_config():
    """Test que la configuration des colonnes requises est correcte"""
    assert isinstance(REQUIRED_COLUMNS, list)
    assert "Avis" in REQUIRED_COLUMNS
    assert "Note" in REQUIRED_COLUMNS

def test_experiment_name_config():
    """Test que la configuration du nom d'expérience est correcte"""
    assert isinstance(INGESTION_EXPERIMENT_NAME, str)
    assert len(INGESTION_EXPERIMENT_NAME) > 0

@patch('mlflow.tensorflow.load_model')  # Mock : évite de charger un vrai modèle MLflow
def test_load_model_from_registry(mock_load_model):
    """Test le chargement d'un modèle depuis le registre MLflow"""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    # Test avec un modèle et une version spécifiques
    model = load_model_from_registry("test_model", version="1")
    assert model == mock_model
    mock_load_model.assert_called_once()
    
    # Reset mock pour le test suivant
    mock_load_model.reset_mock()
    
    # Test avec la dernière version
    with patch('src.utils.get_latest_registered_version', return_value="2"):  # Mock : simule la récupération de la dernière version
        model = load_model_from_registry("test_model")
        assert model == mock_model
        mock_load_model.assert_called_once()

@patch('mlflow.tracking.MlflowClient')  # Mock : évite de se connecter à un vrai serveur MLflow
def test_get_latest_registered_version(mock_client_class):
    """Test la récupération de la dernière version d'un modèle enregistré"""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    # Configure le mock pour retourner des versions de modèle
    mock_version1 = MagicMock(version="1", current_stage="Archived")
    mock_version2 = MagicMock(version="2", current_stage="Production")
    mock_version3 = MagicMock(version="3", current_stage="Staging")
    mock_client.search_model_versions.return_value = [mock_version1, mock_version2, mock_version3]
    
    # Test la récupération de la dernière version (par défaut)
    version = get_latest_registered_version("test_model")
    assert version == "3"
    
    # Test avec filtre sur le stage
    version = get_latest_registered_version("test_model", stage="Production")
    assert version == "2"
    
    # Test si aucune version n'est trouvée
    mock_client.search_model_versions.return_value = []
    with pytest.raises(ValueError):
        get_latest_registered_version("nonexistent_model")

def test_logger_config():
    """Test que la configuration du logger fonctionne correctement"""
    from src.logger_config import get_logger
    
    # Test de création de différents loggers
    data_logger = get_logger('data_ingestion')
    api_logger = get_logger('api')
    
    assert data_logger.name == 'data_ingestion'
    assert api_logger.name == 'api'
    
    # Vérifier que les attributs nécessaires sont présents
    for logger in [data_logger, api_logger]:
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'debug')

def test_model_validation_config():
    """Test que la configuration de validation du modèle est correcte"""
    from src.config import VALIDATION_THRESHOLD
    
    assert isinstance(VALIDATION_THRESHOLD, float)
    assert 0 <= VALIDATION_THRESHOLD <= 1  # Doit être une valeur de probabilité
