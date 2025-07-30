"""
Tests unitaires pour les fonctionnalités de validation de datasets.
Ce fichier se concentre uniquement sur les fonctionnalités de la classe DataIngestionPipeline
liées à la validation des jeux de données.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data_ingestion import DataIngestionPipeline
import mlflow
import tempfile
import os

@pytest.fixture
def mock_validation_dataset():
    """Jeu de données simple pour les tests de validation"""
    return pd.DataFrame({
        'Avis': [
            "Produit excellent pour validation",
            "Qualité médiocre pour validation",
            "Test de validation"
        ],
        'Note': [5, 2, 4]
    })

@pytest.fixture
def temp_validation_csv():
    """Crée un fichier CSV temporaire pour les tests"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        df = pd.DataFrame({
            'Avis': ["Validation avis 1", "Validation avis 2"],
            'Note': [5, 2]
        })
        df.to_csv(tmp.name, index=False)
        yield tmp.name
    os.unlink(tmp.name)

def test_validation_dataset_initialization():
    """Vérifie que le type de dataset est correctement initialisé"""
    # Patch simplifié, seulement les imports nécessaires
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        
        # Test pour jeu de validation
        pipeline = DataIngestionPipeline("test_data.csv", is_validation_set=True)
        assert pipeline.dataset_type == "jdd validation"
        
        # Test pour jeu d'entraînement
        pipeline = DataIngestionPipeline("test_data.csv", is_validation_set=False)
        assert pipeline.dataset_type == "jdd entrainement"

def test_validation_dataset_tagging():
    """Vérifie que les tags MLflow sont correctement appliqués"""
    # Mocks minimaux requis pour tester le tagging
    with patch('mlflow.get_experiment_by_name') as mock_get_exp, \
         patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.tracking.MlflowClient') as mock_client_class, \
         patch('src.data_ingestion.DataIngestionPipeline._execute_pipeline') as mock_execute:
        
        # Configuration des mocks
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_execute.return_value = pd.DataFrame({'Avis': ["Test"], 'Note': [5]})
        
        # Test avec jeu de validation
        pipeline = DataIngestionPipeline("test_data.csv", is_validation_set=True)
        pipeline.run_pipeline()
        mock_client.set_tag.assert_any_call("test-run-id", "dataset_type", "jdd validation")
        
        # Test avec jeu d'entraînement
        pipeline = DataIngestionPipeline("test_data.csv", is_validation_set=False)
        pipeline.run_pipeline()
        mock_client.set_tag.assert_any_call("test-run-id", "dataset_type", "jdd entrainement")

def test_validation_data_preprocessing(mock_validation_dataset):
    """Vérifie que le prétraitement des données de validation est correct"""
    # Configuration minimale pour tester le prétraitement
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        pipeline = DataIngestionPipeline("test_data.csv", is_validation_set=True)
        
        # Test du prétraitement avec des données de validation
        processed_data = pipeline.preprocess_data(mock_validation_dataset)
        
        # Vérifications
        assert len(processed_data) == 3
        assert "Avis" in processed_data.columns
        assert "Note" in processed_data.columns
        assert processed_data["Note"].min() >= 0
        assert processed_data["Note"].max() <= 5

def test_run_validation_pipeline(temp_validation_csv):
    """Teste l'exécution complète du pipeline avec un fichier de validation"""
    # Mocks nécessaires pour exécuter le pipeline sans erreurs
    with patch('mlflow.get_experiment_by_name') as mock_get_exp, \
         patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.log_param'), \
         patch('mlflow.log_metrics'), \
         patch('mlflow.log_artifact'), \
         patch('mlflow.tracking.MlflowClient'), \
         patch('matplotlib.pyplot.figure'), \
         patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.close'), \
         patch('os.remove'):
        
        # Configuration des mocks
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        # Exécution du pipeline sur un fichier CSV réel
        pipeline = DataIngestionPipeline(temp_validation_csv, is_validation_set=True)
        result = pipeline.run_pipeline()
        
        # Vérifications des résultats
        assert isinstance(result, pd.DataFrame)
        assert "Avis" in result.columns
        assert "Note" in result.columns
        assert len(result) > 0
