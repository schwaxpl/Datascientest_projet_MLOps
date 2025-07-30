"""
Module de pipeline d'ingestion de données.
Responsable du chargement et du prétraitement des données d'avis clients.
Utilise MLflow pour le tracking des métriques et paramètres.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import mlflow
import time
import uuid
from pathlib import Path
import os
import matplotlib.pyplot as plt
from datetime import datetime
from src.logger_config import get_logger

# Configuration du logger spécifique au module d'ingestion de données
logger = get_logger('data_ingestion')

from src.config import (
    REQUIRED_COLUMNS,
    INGESTION_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI
)

class DataIngestionPipeline:
    def __init__(self, data_path: str, experiment_name: str = INGESTION_EXPERIMENT_NAME, is_validation_set: bool = False):
        """
        Initialise le pipeline d'ingestion de données.
        
        Args:
            data_path (str): Chemin vers le fichier de données
            experiment_name (str): Nom de l'expérience MLflow
            is_validation_set (bool): Si True, les données seront taguées comme "jdd validation"
                                     sinon comme "jdd entrainement"
        """
        # Génération d'un ID unique pour ce pipeline d'ingestion
        self.pipeline_id = str(uuid.uuid4())[:8]
        logger.info(f"[{self.pipeline_id}] Initialisation du pipeline d'ingestion - Fichier: {data_path}")
        
        self.data_path = data_path
        self.required_columns = REQUIRED_COLUMNS
        
        # Définition du type de dataset
        self.dataset_type = "jdd validation" if is_validation_set else "jdd entrainement"
        logger.info(f"[{self.pipeline_id}] Type de jeu de données: {self.dataset_type}")
        
        # Configuration MLflow
        logger.info(f"[{self.pipeline_id}] Configuration de l'expérience MLflow: {experiment_name}")
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        logger.debug(f"[{self.pipeline_id}] Expérience configurée: ID={self.experiment.experiment_id}")
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Vérifie que les données contiennent les colonnes requises.
        
        Args:
            data (pd.DataFrame): Données à valider
            
        Returns:
            bool: True si les données sont valides
        """
        logger.info(f"[{self.pipeline_id}] Validation des données - {len(data)} lignes")
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"[{self.pipeline_id}] Validation échouée - Colonnes manquantes: {missing_columns}")
            raise ValueError(f"Colonnes manquantes: {missing_columns}")
            
        logger.info(f"[{self.pipeline_id}] Validation réussie - Toutes les colonnes requises sont présentes")
        return True

    def get_data_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcule les statistiques des données.
        
        Args:
            data (pd.DataFrame): Données
            
        Returns:
            Dict: Statistiques des données
        """
        logger.info(f"[{self.pipeline_id}] Calcul des statistiques sur {len(data)} lignes")
        
        # Mesure du temps de calcul
        start_time = time.time()
        
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
        
        # Calcul du temps d'exécution
        execution_time = time.time() - start_time
        logger.info(f"[{self.pipeline_id}] Statistiques calculées en {execution_time:.3f}s")
        logger.debug(f"[{self.pipeline_id}] Statistiques: {stats}")
        
        return stats

    def load_data(self) -> pd.DataFrame:
        """
        Charge les données depuis le fichier source.
        
        Returns:
            pd.DataFrame: Données chargées
        """
        logger.info(f"[{self.pipeline_id}] Chargement des données depuis {self.data_path}")
        
        # Vérification de l'existence du fichier
        if not Path(self.data_path).exists():
            logger.error(f"[{self.pipeline_id}] Fichier introuvable: {self.data_path}")
            raise FileNotFoundError(f"Le fichier {self.data_path} n'existe pas")
        
        # Mesure du temps de chargement
        start_time = time.time()
        
        try:
            data = pd.read_csv(self.data_path)
            load_time = time.time() - start_time
            logger.info(f"[{self.pipeline_id}] Données chargées en {load_time:.3f}s - {len(data)} lignes")
            
            # Validation des données
            self.validate_data(data)
            
            return data
        except Exception as e:
            logger.error(f"[{self.pipeline_id}] Erreur lors du chargement des données: {str(e)}", exc_info=True)
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prétraite les données.
        
        Args:
            data (pd.DataFrame): Données brutes
            
        Returns:
            pd.DataFrame: Données prétraitées
        """
        logger.info(f"[{self.pipeline_id}] Début du prétraitement des données - {len(data)} lignes")
        
        # Mesure du temps de prétraitement
        start_time = time.time()
        
        try:
            processed_data = data.copy()
            
            # Nettoyage des Avis
            logger.debug(f"[{self.pipeline_id}] Nettoyage des avis")
            processed_data["Avis"] = processed_data["Avis"].astype(str).str.strip()
            processed_data["Avis"] = processed_data["Avis"].replace(r'^\s*$', np.nan, regex=True)
            
            # Validation des Notes
            logger.debug(f"[{self.pipeline_id}] Validation des notes")
            initial_note_count = processed_data["Note"].count()
            processed_data["Note"] = pd.to_numeric(processed_data["Note"], errors="coerce")
            converted_note_count = processed_data["Note"].count()
            
            if initial_note_count != converted_note_count:
                logger.warning(f"[{self.pipeline_id}] {initial_note_count - converted_note_count} notes non numériques ont été converties en NaN")
            
            # Suppression des lignes avec des valeurs manquantes
            logger.debug(f"[{self.pipeline_id}] Suppression des lignes avec valeurs manquantes")
            initial_rows = len(processed_data)
            processed_data = processed_data.dropna(subset=["Avis", "Note"])
            dropped_rows = initial_rows - len(processed_data)
            
            if dropped_rows > 0:
                logger.info(f"[{self.pipeline_id}] {dropped_rows} lignes supprimées pour valeurs manquantes ({dropped_rows/initial_rows:.2%})")
            
            # Log des statistiques de prétraitement
            stats = self.get_data_stats(processed_data)
            
            # Calcul du temps de prétraitement
            execution_time = time.time() - start_time
            logger.info(f"[{self.pipeline_id}] Prétraitement terminé en {execution_time:.3f}s - {len(processed_data)} lignes conservées")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"[{self.pipeline_id}] Erreur lors du prétraitement: {str(e)}", exc_info=True)
            raise
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Exécute le pipeline complet d'ingestion de données.
        
        Returns:
            pd.DataFrame: Données prétraitées
        """
        logger.info(f"[{self.pipeline_id}] Démarrage du pipeline d'ingestion")
        start_time = time.time()
        
        with mlflow.start_run(experiment_id=self.experiment.experiment_id) as run:
            logger.info(f"[{self.pipeline_id}] Run MLflow démarré: {run.info.run_id}")
            try:
                # Chargement des données
                logger.info(f"[{self.pipeline_id}] Étape 1: Chargement des données")
                data = self.load_data()
                initial_stats = self.get_data_stats(data)
                
                # Log des paramètres
                logger.debug(f"[{self.pipeline_id}] Enregistrement des paramètres dans MLflow")
                mlflow.log_param("data_path", self.data_path)
                mlflow.log_param("initial_rows", initial_stats["n_rows"])
                mlflow.log_param("pipeline_id", self.pipeline_id)
                mlflow.log_param("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                
                # Prétraitement
                logger.info(f"[{self.pipeline_id}] Étape 2: Prétraitement des données")
                processed_data = self.preprocess_data(data)
                final_stats = self.get_data_stats(processed_data)
                
                # Log des métriques
                logger.debug(f"[{self.pipeline_id}] Enregistrement des métriques dans MLflow")
                removed_rows = initial_stats["n_rows"] - final_stats["n_rows"]
                removed_pct = (removed_rows / initial_stats["n_rows"]) * 100 if initial_stats["n_rows"] > 0 else 0
                
                mlflow.log_metrics({
                    "final_rows": final_stats["n_rows"],
                    "removed_rows": removed_rows,
                    "removed_pct": removed_pct,
                    "avg_note": final_stats["avg_note"],
                    "avg_avis_length": final_stats["avg_avis_length"]
                })
                
                # Sauvegarde temporaire des données pour MLflow
                logger.debug(f"[{self.pipeline_id}] Sauvegarde temporaire des données pour MLflow")
                temp_input_path = f"temp_input_data_{self.pipeline_id}.csv"
                temp_output_path = f"temp_processed_data_{self.pipeline_id}.csv"
                
                data.to_csv(temp_input_path, index=False)
                processed_data.to_csv(temp_output_path, index=False)
                
                # Log des données dans MLflow
                logger.debug(f"[{self.pipeline_id}] Enregistrement des artifacts dans MLflow")
                mlflow.log_artifact(temp_input_path, "data_input")
                mlflow.log_artifact(temp_output_path, "data_processed")
                
                # Tag du jeu de données
                logger.info(f"[{self.pipeline_id}] Application du tag '{self.dataset_type}' au run")
                client = mlflow.tracking.MlflowClient()
                client.set_tag(run.info.run_id, "dataset_type", self.dataset_type)
                client.set_tag(run.info.run_id, "dataset_rows", str(final_stats["n_rows"]))
                client.set_tag(run.info.run_id, "dataset_version", datetime.now().strftime("%Y%m%d_%H%M%S"))
                
                # Log des distributions sous forme de visualisations
                logger.info(f"[{self.pipeline_id}] Étape 3: Génération de visualisations")
                import matplotlib.pyplot as plt
                
                # Distribution des notes
                logger.debug(f"[{self.pipeline_id}] Création de la distribution des notes")
                plt.figure(figsize=(10, 6))
                processed_data['Note'].hist()
                plt.title('Distribution des Notes')
                plt.xlabel('Note')
                plt.ylabel('Fréquence')
                notes_viz_path = f"notes_distribution_{self.pipeline_id}.png"
                plt.savefig(notes_viz_path)
                mlflow.log_artifact(notes_viz_path, "visualizations")
                plt.close()
                
                # Distribution de la longueur des avis
                logger.debug(f"[{self.pipeline_id}] Création de la distribution des longueurs d'avis")
                plt.figure(figsize=(10, 6))
                processed_data['Avis'].str.len().hist()
                plt.title('Distribution de la longueur des avis')
                plt.xlabel('Longueur du texte')
                plt.ylabel('Fréquence')
                avis_viz_path = f"avis_length_distribution_{self.pipeline_id}.png"
                plt.savefig(avis_viz_path)
                mlflow.log_artifact(avis_viz_path, "visualizations")
                plt.close()
                
                # Nettoyage des fichiers temporaires
                logger.debug(f"[{self.pipeline_id}] Nettoyage des fichiers temporaires")
                try:
                    os.remove(temp_input_path)
                    os.remove(temp_output_path)
                    os.remove(notes_viz_path)
                    os.remove(avis_viz_path)
                except Exception as e:
                    logger.warning(f"[{self.pipeline_id}] Erreur lors du nettoyage des fichiers temporaires: {str(e)}")
                
                # Calcul du temps total d'exécution
                total_time = time.time() - start_time
                logger.info(f"[{self.pipeline_id}] Pipeline d'ingestion terminé avec succès en {total_time:.3f}s")
                mlflow.log_metric("pipeline_execution_time", total_time)
                
                return processed_data
                
            except Exception as e:
                logger.error(f"[{self.pipeline_id}] Erreur pendant l'ingestion: {str(e)}", exc_info=True)
                mlflow.log_param("error", str(e))
                mlflow.log_param("error_type", type(e).__name__)
                raise
