"""
Module d'entraînement du modèle.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Optional
import os
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def get_ingestion_data(run_id: Optional[str] = None) -> str:
    """
    Récupère les données ingérées depuis MLflow.
    
    Args:
        run_id (Optional[str]): ID du run MLflow contenant les données.
                              Si None, utilise le dernier run réussi.
    
    Returns:
        str: Chemin vers le fichier de données traitées
    """
    client = MlflowClient()
    
    if run_id is None:
        # Recherche du dernier run réussi de l'expérience data_ingestion_api
        experiment = mlflow.get_experiment_by_name("data_ingestion_api")
        if not experiment:
            raise ValueError("Aucune expérience d'ingestion de données trouvée")
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            raise ValueError("Aucun run d'ingestion trouvé")
            
        run_id = runs[0].info.run_id
    
    # Téléchargement des artifacts du run spécifié
    artifacts_dir = client.download_artifacts(run_id, "data_processed")
    csv_files = [f for f in os.listdir(artifacts_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"Aucun fichier CSV trouvé dans les artifacts du run {run_id}")
        
    return os.path.join(artifacts_dir, csv_files[0])

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