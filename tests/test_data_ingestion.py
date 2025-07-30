"""
Tests unitaires pour le pipeline d'ingestion de données.
Ce fichier teste les fonctionnalités de base de la classe DataIngestionPipeline
sans dépendance à l'API FastAPI.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_ingestion import DataIngestionPipeline
from unittest.mock import patch, MagicMock
import tempfile
import os

@pytest.fixture
def sample_data():
    """Jeu de données simple avec différents cas de test (valeurs vides, NaN)"""
    return pd.DataFrame({
        'Avis': [
            "Excellent produit",
            "Produit moyen",
            "",  # Chaîne vide
            "Très satisfait",
            np.nan  # Valeur manquante
        ],
        'Note': [5, 3, 2, 4, 1]
    })

@pytest.fixture
def temp_csv():
    """Crée un fichier CSV temporaire pour les tests"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        df = pd.DataFrame({
            'Avis': ["Test avis", "Autre avis"],
            'Note': [5, 4]
        })
        df.to_csv(tmp.name, index=False)
        yield tmp.name
    os.unlink(tmp.name)

def test_data_ingestion_pipeline_initialization():
    """Vérifie l'initialisation correcte du pipeline d'ingestion"""
    # Mock MLflow pour éviter les appels externes
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        
        # Initialisation du pipeline
        pipeline = DataIngestionPipeline("test_data.csv")
        
        # Vérifications des propriétés
        assert pipeline.data_path == "test_data.csv"
        assert pipeline.required_columns == ["Avis", "Note"]
        assert pipeline.dataset_type == "jdd entrainement"  # Valeur par défaut

def test_validate_data(sample_data):
    """Vérifie que la validation des données fonctionne pour un DataFrame valide"""
    # Mock MLflow minimalement requis
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        
        pipeline = DataIngestionPipeline("test.csv")
        
        # Test que la validation passe
        assert pipeline.validate_data(sample_data) == True

def test_validate_data_missing_columns():
    """Vérifie que la validation échoue quand il manque des colonnes requises"""
    # Mock MLflow minimalement requis
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        
        pipeline = DataIngestionPipeline("test.csv")
        invalid_data = pd.DataFrame({
            'Texte': ["Test"],  # Pas la colonne "Avis"
            'Score': [5]        # Pas la colonne "Note"
        })
        
        # Test que la validation échoue avec ValueError
        with pytest.raises(ValueError) as exc_info:
            pipeline.validate_data(invalid_data)
        assert "Colonnes manquantes" in str(exc_info.value)

def test_get_data_stats(sample_data):
    """Vérifie que les statistiques sont correctement calculées"""
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        pipeline = DataIngestionPipeline("test.csv")
        
        # Obtention des statistiques
        stats = pipeline.get_data_stats(sample_data)
        
        # Vérifications
        assert isinstance(stats, dict)
        assert stats["n_rows"] == 5
        assert stats["n_missing_avis"] == 1
        assert stats["n_missing_notes"] == 0
        assert stats["avg_note"] == 3.0
        assert stats["min_note"] == 1
        assert stats["max_note"] == 5
        assert "avg_avis_length" in stats

def test_load_data(temp_csv):
    """Vérifie le chargement d'un fichier CSV valide"""
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        pipeline = DataIngestionPipeline(temp_csv)
        
        # Chargement des données
        data = pipeline.load_data()
        
        # Vérifications
        assert isinstance(data, pd.DataFrame)
        assert all(col in data.columns for col in ["Avis", "Note"])

def test_load_data_file_not_found():
    """Vérifie qu'une exception est levée quand le fichier n'existe pas"""
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        pipeline = DataIngestionPipeline("nonexistent.csv")
        
        # Vérification que FileNotFoundError est levé
        with pytest.raises(FileNotFoundError):
            pipeline.load_data()

def test_preprocess_data(sample_data):
    """Vérifie le prétraitement des données"""
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        pipeline = DataIngestionPipeline("test.csv")
        
        # Prétraitement des données
        processed_data = pipeline.preprocess_data(sample_data)
        
        # Vérifications
        assert len(processed_data) < len(sample_data)  # Lignes invalides supprimées
        assert processed_data["Avis"].isna().sum() == 0  # Pas de valeurs manquantes
        assert processed_data["Note"].isna().sum() == 0  # Pas de valeurs manquantes
        assert all(isinstance(note, (int, float)) for note in processed_data["Note"])

def test_run_pipeline_with_mlflow(temp_csv):
    """Vérifie l'exécution complète du pipeline avec MLflow"""
    # Ce test vérifie que run_pipeline() interagit correctement avec MLflow
    # Mais on évite d'exécuter réellement MLflow pour éviter les problèmes de fichiers YAML
    
    # Mock complet pour MLflow
    with patch('mlflow.start_run', create=True) as mock_start_run, \
         patch('mlflow.get_experiment_by_name') as mock_get_exp, \
         patch('mlflow.log_param') as mock_log_param, \
         patch('mlflow.log_metrics') as mock_log_metrics, \
         patch('mlflow.log_artifact') as mock_log_artifact, \
         patch('mlflow.tracking.MlflowClient') as mock_client_class, \
         patch('mlflow.tracking', create=True), \
         patch('matplotlib.pyplot', create=True), \
         patch('pandas.DataFrame.to_csv'), \
         patch('os.remove'), \
         patch('os.path.exists', return_value=True):
        
        # Configuration des mocks
        mock_get_exp.return_value = MagicMock(experiment_id="test-exp-id")
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id" 
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Créer une instance du pipeline avec le fichier temporaire
        pipeline = DataIngestionPipeline(temp_csv)
        
        # Mock de méthodes internes pour isoler le test
        with patch.object(pipeline, 'load_data') as mock_load_data, \
             patch.object(pipeline, 'preprocess_data') as mock_preprocess:
            
            # Configurer les mocks pour retourner des données attendues
            mock_data = pd.DataFrame({"Avis": ["test1", "test2"], "Note": [5, 3]})
            mock_processed = pd.DataFrame({"Avis": ["test1", "test2"], "Note": [5, 3]})
            
            mock_load_data.return_value = mock_data
            mock_preprocess.return_value = mock_processed
            
            # Exécuter le pipeline
            result = pipeline.run_pipeline()
            
            # Vérifications
            assert result is mock_processed  # Vérifier que c'est bien le DataFrame prétraité
            mock_load_data.assert_called_once()
            mock_preprocess.assert_called_once_with(mock_data)
            assert mock_log_param.called
            assert mock_log_metrics.called
            assert mock_log_artifact.called

def test_run_pipeline_error_handling():
    """Vérifie que les erreurs sont correctement gérées lors de l'exécution du pipeline"""
    with patch('mlflow.get_experiment_by_name') as mock_get_exp, \
         patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.log_param'):
        
        # Configuration des mocks
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        mock_start_run.return_value.__enter__.return_value = MagicMock()
        
        # Création du pipeline avec un fichier inexistant
        pipeline = DataIngestionPipeline("nonexistent.csv")
        
        # Vérification que l'exception appropriée est levée
        with pytest.raises(FileNotFoundError):
            pipeline.run_pipeline()
