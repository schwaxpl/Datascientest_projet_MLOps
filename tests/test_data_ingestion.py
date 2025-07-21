"""
Tests unitaires pour le pipeline d'ingestion de données.
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
    return pd.DataFrame({
        'Avis': [
            "Excellent produit",
            "Produit moyen",
            "",
            "Très satisfait",
            np.nan
        ],
        'Note': [5, 3, 2, 4, 1]
    })

@pytest.fixture
def temp_csv():
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        df = pd.DataFrame({
            'Avis': ["Test avis", "Autre avis"],
            'Note': [5, 4]
        })
        df.to_csv(tmp.name, index=False)
        yield tmp.name
    os.unlink(tmp.name)

def test_data_ingestion_pipeline_initialization():
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        pipeline = DataIngestionPipeline("test_data.csv")
        assert pipeline.data_path == "test_data.csv"
        assert pipeline.required_columns == ["Avis", "Note"]

def test_validate_data(sample_data):
    pipeline = DataIngestionPipeline("test.csv")
    assert pipeline.validate_data(sample_data) == True

def test_validate_data_missing_columns():
    pipeline = DataIngestionPipeline("test.csv")
    invalid_data = pd.DataFrame({
        'Texte': ["Test"],
        'Score': [5]
    })
    with pytest.raises(ValueError) as exc_info:
        pipeline.validate_data(invalid_data)
    assert "Colonnes manquantes" in str(exc_info.value)

def test_get_data_stats(sample_data):
    pipeline = DataIngestionPipeline("test.csv")
    stats = pipeline.get_data_stats(sample_data)
    
    assert isinstance(stats, dict)
    assert stats["n_rows"] == 5
    assert stats["n_missing_avis"] == 1
    assert stats["n_missing_notes"] == 0
    assert stats["avg_note"] == 3.0
    assert stats["min_note"] == 1
    assert stats["max_note"] == 5

def test_load_data(temp_csv):
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        pipeline = DataIngestionPipeline(temp_csv)
        data = pipeline.load_data()
        
        assert isinstance(data, pd.DataFrame)
        assert all(col in data.columns for col in ["Avis", "Note"])

def test_load_data_file_not_found():
    pipeline = DataIngestionPipeline("nonexistent.csv")
    with pytest.raises(FileNotFoundError):
        pipeline.load_data()

def test_preprocess_data(sample_data):
    pipeline = DataIngestionPipeline("test.csv")
    processed_data = pipeline.preprocess_data(sample_data)
    
    assert len(processed_data) < len(sample_data)  # Vérification de la suppression des lignes invalides
    assert processed_data["Avis"].isna().sum() == 0  # Pas de valeurs manquantes
    assert processed_data["Note"].isna().sum() == 0
    assert all(isinstance(note, (int, float)) for note in processed_data["Note"])

@patch('mlflow.start_run')
@patch('mlflow.log_param')
@patch('mlflow.log_metrics')
def test_run_pipeline_with_mlflow(mock_log_metrics, mock_log_param, mock_start_run, temp_csv):
    mock_start_run.return_value.__enter__.return_value = MagicMock()
    
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        pipeline = DataIngestionPipeline(temp_csv)
        result = pipeline.run_pipeline()
        
        assert isinstance(result, pd.DataFrame)
        mock_log_param.assert_called()
        mock_log_metrics.assert_called()

@patch('mlflow.start_run')
def test_run_pipeline_error_handling(mock_start_run):
    mock_start_run.return_value.__enter__.return_value = MagicMock()
    
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        pipeline = DataIngestionPipeline("nonexistent.csv")
        
        with pytest.raises(FileNotFoundError):
            pipeline.run_pipeline()
