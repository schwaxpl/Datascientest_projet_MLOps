"""
Fonctions utilitaires partagées entre les différents modules.
"""

import mlflow
from mlflow.tracking import MlflowClient
import tensorflow as tf
import pickle
import os
import boto3
from typing import Optional, Dict
import pandas as pd


def get_mlflow_client():
    """
    Crée et retourne un client MLflow configuré pour fonctionner avec MinIO/S3.
    Configure également les variables d'environnement S3 nécessaires.
    
    Returns:
        MlflowClient: Un client MLflow configuré
    """
    # S'assurer que les variables d'environnement sont définies
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Configuration pour S3/MinIO
    if "MLFLOW_S3_ENDPOINT_URL" in os.environ:
        # La configuration existe déjà, vérifier que boto3 est correctement configuré
        boto3_session = boto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
        )
        
    return MlflowClient(tracking_uri=mlflow_tracking_uri)

def get_latest_model_version(client: MlflowClient, model_name: str):
    """
    Récupère la dernière version du modèle en privilégiant les versions en production/staging.
    
    Args:
        client: MLflow client
        model_name: Nom du modèle à rechercher
    
    Returns:
        La dernière version du modèle selon la stratégie: Production > Staging > Latest
    """
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if not model_versions:
        raise ValueError(f"Aucune version du modèle {model_name} trouvée dans MLflow")
    
    # Trier par version décroissante et statut
    production_versions = [mv for mv in model_versions if mv.current_stage == 'Production']
    staging_versions = [mv for mv in model_versions if mv.current_stage == 'Staging']
    
    if production_versions:
        return production_versions[0]  # Dernière version en production
    elif staging_versions:
        return staging_versions[0]  # Dernière version en staging
    else:
        return model_versions[0]  # Dernière version disponible

def load_model_from_registry(model_name: str, version: Optional[str] = None) -> tf.keras.Model:
    """
    Charge un modèle depuis MLflow model registry avec gestion des différents formats.
    
    Args:
        model_name: Nom du modèle à charger
        version: Version spécifique à charger. Si None, utilise la dernière version disponible
    
    Returns:
        Le modèle chargé
    """
    client = MlflowClient()
    if version:
        model_version = next(
            (mv for mv in client.search_model_versions(f"name='{model_name}'")
             if mv.version == version),
            None
        )
        if not model_version:
            raise ValueError(f"Version {version} non trouvée pour le modèle {model_name}")
    else:
        model_version = get_latest_model_version(client, model_name)
    
    try:
        model = mlflow.tensorflow.load_model(f"models:/{model_name}/{model_version.version}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle tensorflow: {str(e)}")
        print("Tentative de chargement avec keras...")
        model = mlflow.keras.load_model(f"models:/{model_name}/{model_version.version}")
    
    return model

def get_latest_run_artifact(experiment_name: str, artifact_path: str, filter_string: str = "status = 'FINISHED'") -> str:
    """
    Récupère un artifact depuis le dernier run d'une expérience.
    Si l'artifact est lié à un modèle ou vectorizer, privilégie l'utilisation
    des fichiers locaux plutôt que de télécharger depuis MLflow.
    
    Args:
        experiment_name: Nom de l'expérience MLflow
        artifact_path: Chemin de l'artifact à récupérer
        filter_string: Filtre pour la recherche des runs
    
    Returns:
        str: Chemin local vers l'artifact (local ou téléchargé depuis MLflow)
    """
    # Vérifier si l'artifact est un modèle ou vectorizer local
    from src.config import MODEL_PATH, VECTORIZER_PATH
    
    if artifact_path.endswith("vectorizer.pkl") and os.path.exists(VECTORIZER_PATH):
        return VECTORIZER_PATH
    elif artifact_path.endswith("model.pkl") and os.path.exists(MODEL_PATH):
        return MODEL_PATH
    
    # Si ce n'est pas un fichier local connu, télécharger depuis MLflow
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Expérience {experiment_name} non trouvée")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"Aucun run trouvé pour l'expérience {experiment_name}")
    
    # Télécharger l'artifact depuis MLflow directement au chemin de destination
    return client.download_artifacts(runs[0].info.run_id, artifact_path)

def get_vectorizer_from_run(run_id: str, vectorizer_filename: str) -> object:
    """
    Récupère le vectorizer depuis un run MLflow ou depuis le fichier local.
    Privilégie l'utilisation du fichier local dans /app/models/ au lieu de 
    télécharger depuis MLflow.
    
    Args:
        run_id: ID du run MLflow (ignoré si le fichier local est disponible)
        vectorizer_filename: Nom du fichier du vectorizer
    
    Returns:
        Le vectorizer chargé
    """
    from src.config import VECTORIZER_PATH
    
    # Privilégier le vectorizer local
    try:
        with open(VECTORIZER_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Vectorizer local non disponible: {str(e)}")
        print("Tentative de récupération depuis MLflow...")
        
        # Si le vectorizer local n'est pas disponible, essayer de le récupérer depuis MLflow
        client = MlflowClient()
        local_path = client.download_artifacts(run_id, vectorizer_filename)
        
        with open(local_path, 'rb') as f:
            return pickle.load(f)

def get_latest_registered_version(client: MlflowClient, model_name: str):
    """
    Récupère la dernière version créée pour un modèle.
    
    Args:
        client: MLflow client
        model_name: Nom du modèle
    
    Returns:
        La dernière version créée
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"Aucune version trouvée pour le modèle {model_name}")
    
    # Trier par version décroissante
    versions.sort(key=lambda x: int(x.version), reverse=True)
    return versions[0]
