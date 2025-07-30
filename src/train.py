"""
Module d'entraînement du modèle.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Optional, Any
import os
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import time
import uuid
from datetime import datetime
from src.logger_config import get_logger
from src.utils import get_mlflow_client
# Configuration du logger spécifique au module d'entraînement
logger = get_logger('train')

def get_ingestion_data(run_id: Optional[str] = None) -> str:
    """
    Récupère les données ingérées depuis MLflow.
    
    Args:
        run_id (Optional[str]): ID du run MLflow contenant les données.
                              Si None, utilise le dernier run réussi.
    
    Returns:
        str: Chemin vers le fichier de données traitées
    """
    # Génération d'un ID unique pour cette opération
    op_id = str(uuid.uuid4())[:8]
    logger.info(f"[{op_id}] Récupération des données d'ingestion - Run ID: {run_id or 'Auto (dernier run)'}")
    
    
    client = get_mlflow_client()
    
    if run_id is None:
        # Recherche du dernier run réussi de l'expérience data_ingestion_api
        logger.info(f"[{op_id}] Recherche du dernier run d'ingestion réussi")
        experiment = mlflow.get_experiment_by_name("data_ingestion_api")
        if not experiment:
            logger.error(f"[{op_id}] Aucune expérience d'ingestion de données trouvée")
            raise ValueError("Aucune expérience d'ingestion de données trouvée")
        
        logger.debug(f"[{op_id}] Recherche de runs pour l'expérience: {experiment.experiment_id}")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            logger.error(f"[{op_id}] Aucun run d'ingestion trouvé")
            raise ValueError("Aucun run d'ingestion trouvé")
            
        run_id = runs[0].info.run_id
        logger.info(f"[{op_id}] Dernier run trouvé: {run_id}")
    
    # Téléchargement des artifacts du run spécifié
    logger.info(f"[{op_id}] Téléchargement des artifacts du run {run_id}")
    start_time = time.time()
    
    # Récupérer d'abord la liste de tous les artifacts du run
    logger.info(f"[{op_id}] Listing des artifacts disponibles dans le run {run_id}")
    try:
        artifacts = client.list_artifacts(run_id)
        logger.info(f"[{op_id}] Artifacts disponibles dans le run: {[a.path for a in artifacts]}")
        
        # Vérifier si data_processed est présent dans les artefacts
        data_processed_exists = any(a.path == "data_processed" or a.path.startswith("data_processed/") for a in artifacts)
        logger.info(f"[{op_id}] data_processed {'existe' if data_processed_exists else 'n\'existe pas'} dans les artifacts")
    except Exception as e:
        logger.warning(f"[{op_id}] Erreur lors du listing des artifacts: {str(e)}")
        data_processed_exists = False
    
    try:
        # Essayer d'abord le chemin direct comme prévu dans data_ingestion.py
        logger.info(f"[{op_id}] Tentative de téléchargement depuis 'data_processed'...")
        artifacts_dir = client.download_artifacts(run_id, "data_processed")
        logger.info(f"[{op_id}] Artifacts 'data_processed' téléchargés avec succès à: {artifacts_dir}")
        
        # Vérifier le contenu du dossier téléchargé
        if os.path.exists(artifacts_dir) and os.path.isdir(artifacts_dir):
            files_in_dir = os.listdir(artifacts_dir)
            logger.info(f"[{op_id}] Contenu du dossier téléchargé: {files_in_dir}")
            csv_files = [os.path.join(artifacts_dir, f) for f in files_in_dir if f.endswith('.csv')]
            logger.info(f"[{op_id}] Fichiers CSV trouvés: {csv_files}")
        else:
            logger.warning(f"[{op_id}] Le chemin téléchargé n'est pas un dossier valide: {artifacts_dir}")
            csv_files = []
    except Exception as e:
        logger.warning(f"[{op_id}] Erreur lors du téléchargement depuis 'data_processed': {str(e)}")
        logger.info(f"[{op_id}] Tentative de téléchargement depuis la racine...")
        
        try:
            # Plan B: télécharger tous les artefacts et chercher les CSVs
            artifacts_dir = client.download_artifacts(run_id, "")
            logger.info(f"[{op_id}] Tous les artifacts téléchargés dans: {artifacts_dir}")
            
            # Rechercher récursivement tous les fichiers CSV
            csv_files = []
            for root, dirs, files in os.walk(artifacts_dir):
                logger.debug(f"[{op_id}] Parcours de {root}: {files}")
                csv_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
            
            logger.info(f"[{op_id}] Fichiers CSV trouvés après recherche récursive: {csv_files}")
        except Exception as second_e:
            logger.error(f"[{op_id}] Échec également lors du téléchargement depuis la racine: {str(second_e)}")
            raise ValueError(f"Impossible de récupérer les artifacts du run {run_id}: {str(e)} puis {str(second_e)}")
            
    logger.info(f"[{op_id}] Artifacts téléchargés en {time.time() - start_time:.3f}s - Chemin: {artifacts_dir}")
    logger.info(f"[{op_id}] Fichiers CSV trouvés: {csv_files}")
    
    if not csv_files:
        logger.error(f"[{op_id}] Aucun fichier CSV trouvé dans les artifacts du run {run_id}")
        raise ValueError(f"Aucun fichier CSV trouvé dans les artifacts du run {run_id}")
    
    # Utiliser directement le chemin complet du premier CSV trouvé
    data_path = csv_files[0]
    logger.info(f"[{op_id}] Fichier de données trouvé: {data_path}")
    
    return data_path

from src.config import (
    MODEL_NAME,
    VECTORIZER_PATH,
    TRAINING_EXPERIMENT_NAME,
    TRAIN_TEST_SPLIT_RATIO,
    RANDOM_SEED,
    TRAINING_EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    REQUIRED_COLUMNS,
    POSITIVE_REVIEW_THRESHOLD,
    MLFLOW_TRACKING_URI
)

from src.utils import load_model_from_registry, get_latest_registered_version

def train_model(run_id: Optional[str] = None, model_name: Optional[str] = None, base_model_name: Optional[str] = None, base_model_version: Optional[str] = None) -> Dict:
    """
    Entraîne le modèle avec le vectorizer existant et sauvegarde les résultats.
    
    Args:
        run_id (Optional[str]): ID du run MLflow contenant les données d'entraînement.
                              Si None, utilise le dernier run d'ingestion.
        model_name (Optional[str]): Nom sous lequel enregistrer le nouveau modèle.
                                  Si None, utilise MODEL_NAME de la configuration.
        base_model_name (Optional[str]): Nom du modèle à utiliser comme base pour l'entraînement.
                                       Si None, utilise MODEL_NAME de la configuration.
        base_model_version (Optional[str]): Version spécifique du modèle de base à utiliser.
                                          Si None, utilise la dernière version disponible.
    
    Returns:
        Dict: Métriques d'évaluation du modèle
    
    Raises:
        ImportError: Si tensorflow/keras n'est pas installé
        Exception: Pour toute autre erreur pendant l'entraînement
    """
    # Vérification de tensorflow

    # Configuration de MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(TRAINING_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        try:
            # Chargement et préparation des données
            data_path = get_ingestion_data(run_id)
            data = pd.read_csv(data_path)
            
            if not all(col in data.columns for col in REQUIRED_COLUMNS):
                raise ValueError(f"Le fichier doit contenir les colonnes {REQUIRED_COLUMNS}")
            
            # Préparation des features et labels
            y = (data['Note'] > POSITIVE_REVIEW_THRESHOLD).astype(int)
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            X = vectorizer.transform(data['Avis'])
            
            # Split des données
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_SEED, stratify=y
            )
            
            # Chargement et configuration du modèle
            model = load_model_from_registry(
                base_model_name or MODEL_NAME,
                version=base_model_version
            )
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Entraînement
            history = model.fit(
                X_train.toarray(),
                y_train,
                epochs=TRAINING_EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT,
                verbose=1
            )
            
            # Évaluation
            train_metrics = model.evaluate(X_train.toarray(), y_train, verbose=0)
            test_metrics = model.evaluate(X_test.toarray(), y_test, verbose=0)
            
            train_score = train_metrics[1]  # accuracy est le second métrique
            test_score = test_metrics[1]
            
            # Prédictions pour le rapport de classification
            y_pred = np.argmax(model.predict(X_test.toarray()), axis=1)
            
            print(f"Score d'entraînement: {train_score:.3f}")
            print(f"Score de test: {test_score:.3f}")
            print("\nRapport de classification:")
            print(classification_report(y_test, y_pred))
            
            # Log des métriques dans MLflow
            mlflow.log_metrics({
                "train_accuracy": train_score,
                "test_accuracy": test_score
            })
            
            # Log de la matrice de confusion
            conf_matrix = confusion_matrix(y_test, y_pred)
            mlflow.log_metric("true_negatives", conf_matrix[0][0])
            mlflow.log_metric("false_positives", conf_matrix[0][1])
            mlflow.log_metric("false_negatives", conf_matrix[1][0])
            mlflow.log_metric("true_positives", conf_matrix[1][1])
            # Log du F1-score par classe
            classification_dict = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_metric("f1_score_class_0", classification_dict['0']['f1-score'])
            mlflow.log_metric("f1_score_class_1", classification_dict['1']['f1-score'])
            # Log des paramètres
            mlflow.log_param("ingestion_run_id", run_id if run_id else "latest")
            mlflow.log_param("data_path", data_path)
            
            # Log des références du modèle d'origine
            mlflow.log_param("base_model_name", base_model_name or MODEL_NAME)
            mlflow.log_param("base_model_version", base_model_version or "latest")
            
            # Log des métadonnées du dataset
            try:
                dataset_info = {
                    "dataset_source": data_path,
                    "dataset_size": len(data),
                    "dataset_features": list(data.columns),
                    "positive_samples": int(sum(y)),
                    "negative_samples": int(len(y) - sum(y)),
                    "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                mlflow.log_dict(dataset_info, "dataset_info.json")
                logger.info(f"Métadonnées du dataset loggées: {len(data)} échantillons")
            except Exception as e:
                logger.warning(f"Erreur lors du logging des métadonnées du dataset: {str(e)}")
            
            # Log et enregistrement du modèle dans MLflow
            final_model_name = model_name or MODEL_NAME
            mlflow.keras.log_model(
                model,
                "model",
                registered_model_name=final_model_name
            )
            
            # Récupérer la dernière version créée
            client = MlflowClient()
            latest_version = get_latest_registered_version(client, final_model_name)
            
            # Mettre le tag "à valider" pour ce nouveau modèle et utiliser un alias au lieu d'un stage
            logger.info(f"Application du tag 'à valider' pour la version {latest_version.version}")
            
            # Set tag to indicate model needs validation
            client.set_model_version_tag(
                name=final_model_name,
                version=latest_version.version,
                key="status",
                value="à valider"
            )
            
            # Use alias instead of stage (recommended migration path)
            client.set_registered_model_alias(
                name=final_model_name,
                alias="staging",
                version=latest_version.version
            )
            logger.info(f"Version {latest_version.version} marquée comme 'à valider' et avec alias 'staging'")
            
            # Sauvegarde locale optionnelle
            #model.save('models/tf_idf_mdl.pkl')
            
            return {
                "train_accuracy": train_score,
                "test_accuracy": test_score,
                "data_path": data_path,
                "run_id": mlflow.active_run().info.run_id,
                "model_name": final_model_name,
                "model_version": latest_version.version
            }
                
        except Exception as e:
            # Log l'erreur dans MLflow
            mlflow.log_param("error", str(e))
            print(f"Erreur pendant l'entraînement: {str(e)}")
            # On peut aussi logger la stack trace complète
            import traceback
            mlflow.log_text(traceback.format_exc(), "error_trace.txt")
            # Re-raise l'exception pour que l'appelant sache qu'il y a eu une erreur
            raise