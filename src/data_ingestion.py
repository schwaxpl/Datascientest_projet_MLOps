"""
Module de pipeline d'ingestion de données.
Responsable du chargement et du prétraitement des données d'avis clients.
Utilise MLflow pour le tracking des métriques et paramètres.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import mlflow
import logging
from pathlib import Path
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import (
    REQUIRED_COLUMNS,
    INGESTION_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI
)

class DataIngestionPipeline:
    def __init__(self, data_path: str, experiment_name: str = INGESTION_EXPERIMENT_NAME):
        """
        Initialise le pipeline d'ingestion de données.
        
        Args:
            data_path (str): Chemin vers le fichier de données
            experiment_name (str): Nom de l'expérience MLflow
        """
        self.data_path = data_path
        self.required_columns = REQUIRED_COLUMNS
        
        # Configuration MLflow
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Vérifie que les données contiennent les colonnes requises.
        
        Args:
            data (pd.DataFrame): Données à valider
            
        Returns:
            bool: True si les données sont valides
        """
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes: {missing_columns}")
        return True

    def get_data_stats(self, data: pd.DataFrame) -> Dict:
        """
        Calcule les statistiques des données.
        
        Args:
            data (pd.DataFrame): Données
            
        Returns:
            Dict: Statistiques des données
        """
        # Conversion des types numpy en types Python standard
        stats = {
            "n_rows": int(len(data)),
            "n_missing_avis": int(data["Avis"].isna().sum()),
            "n_missing_notes": int(data["Note"].isna().sum()),
            "avg_note": float(data["Note"].mean()),
            "min_note": int(data["Note"].min()),
            "max_note": int(data["Note"].max()),
            "avg_avis_length": float(data["Avis"].str.len().mean())
        }
        return stats

    def load_data(self) -> pd.DataFrame:
        """
        Charge les données depuis le fichier source.
        
        Returns:
            pd.DataFrame: Données chargées
        """
        logger.info(f"Chargement des données depuis {self.data_path}")
        
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Le fichier {self.data_path} n'existe pas")
            
        data = pd.read_csv(self.data_path)
        self.validate_data(data)
        
        return data
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prétraite les données.
        
        Args:
            data (pd.DataFrame): Données brutes
            
        Returns:
            pd.DataFrame: Données prétraitées
        """
        logger.info("Prétraitement des données")
        processed_data = data.copy()
        
        # Nettoyage des Avis
        processed_data["Avis"] = processed_data["Avis"].astype(str).str.strip()
        processed_data["Avis"] = processed_data["Avis"].replace(r'^\s*$', np.nan, regex=True)
        
        # Validation des Notes
        processed_data["Note"] = pd.to_numeric(processed_data["Note"], errors="coerce")
        
        # Suppression des lignes avec des valeurs manquantes
        processed_data = processed_data.dropna(subset=["Avis", "Note"])
        
        # Log des statistiques de prétraitement
        stats = self.get_data_stats(processed_data)
        logger.info(f"Statistiques après prétraitement: {stats}")
        
        return processed_data
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Exécute le pipeline complet d'ingestion de données.
        
        Returns:
            pd.DataFrame: Données prétraitées
        """
        with mlflow.start_run(experiment_id=self.experiment.experiment_id) as run:
            try:
                # Chargement des données
                data = self.load_data()
                initial_stats = self.get_data_stats(data)
                
                # Log des paramètres
                mlflow.log_param("data_path", self.data_path)
                mlflow.log_param("initial_rows", initial_stats["n_rows"])
                
                # Prétraitement
                processed_data = self.preprocess_data(data)
                final_stats = self.get_data_stats(processed_data)
                
                # Log des métriques
                mlflow.log_metrics({
                    "final_rows": final_stats["n_rows"],
                    "removed_rows": initial_stats["n_rows"] - final_stats["n_rows"],
                    "avg_note": final_stats["avg_note"],
                    "avg_avis_length": final_stats["avg_avis_length"]
                })
                
                # Sauvegarde temporaire des données pour MLflow
                temp_input_path = "temp_input_data.csv"
                temp_output_path = "temp_processed_data.csv"
                
                data.to_csv(temp_input_path, index=False)
                processed_data.to_csv(temp_output_path, index=False)
                
                # Log des données dans MLflow
                mlflow.log_artifact(temp_input_path, "data_input")
                mlflow.log_artifact(temp_output_path, "data_processed")
                
                # Log des distributions sous forme de visualisations
                import matplotlib.pyplot as plt
                
                # Distribution des notes
                plt.figure(figsize=(10, 6))
                processed_data['Note'].hist()
                plt.title('Distribution des Notes')
                plt.xlabel('Note')
                plt.ylabel('Fréquence')
                plt.savefig('notes_distribution.png')
                mlflow.log_artifact('notes_distribution.png', "visualizations")
                plt.close()
                
                # Distribution de la longueur des avis
                plt.figure(figsize=(10, 6))
                processed_data['Avis'].str.len().hist()
                plt.title('Distribution de la longueur des avis')
                plt.xlabel('Longueur du texte')
                plt.ylabel('Fréquence')
                plt.savefig('avis_length_distribution.png')
                mlflow.log_artifact('avis_length_distribution.png', "visualizations")
                plt.close()
                
                # Nettoyage des fichiers temporaires
                os.remove(temp_input_path)
                os.remove(temp_output_path)
                os.remove('notes_distribution.png')
                os.remove('avis_length_distribution.png')
                
                logger.info("Pipeline d'ingestion terminé avec succès")
                return processed_data
                
            except Exception as e:
                logger.error(f"Erreur pendant l'ingestion: {str(e)}")
                mlflow.log_param("error", str(e))
                raise
