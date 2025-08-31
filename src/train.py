"""
Module d'entraînement du modèle.
"""

import pickle
import pandas as pd
import numpy as np
import sys
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

def diagnose_csv_file(file_path: str) -> Dict[str, Any]:
    """
    Analyse un fichier CSV pour identifier d'éventuels problèmes.
    
    Args:
        file_path (str): Chemin vers le fichier CSV à analyser
    
    Returns:
        Dict[str, Any]: Informations de diagnostic sur le fichier
    """
    diagnostics = {
        "file_exists": os.path.exists(file_path),
        "file_size": 0,
        "file_size_mb": 0,
        "num_lines": 0,
        "line_issues": [],
        "encoding_guess": None,
        "first_lines": [],
        "delimiters_found": {},
        "status": "unknown"
    }
    
    if not diagnostics["file_exists"]:
        diagnostics["status"] = "error"
        diagnostics["message"] = f"Le fichier {file_path} n'existe pas"
        return diagnostics
    
    # Taille du fichier
    file_size = os.path.getsize(file_path)
    diagnostics["file_size"] = file_size
    diagnostics["file_size_mb"] = file_size / (1024 * 1024)
    
    # Vérifier l'encodage (méthode simplifiée sans dépendances externes)
    encodings_to_try = ['utf-8', 'latin1', 'cp1252']
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                f.read(1000)  # Lire un échantillon
                diagnostics["encoding_guess"] = {"encoding": enc, "confidence": 1.0}
                break
        except UnicodeDecodeError:
            continue
    else:
        diagnostics["encoding_guess"] = {"encoding": "unknown", "confidence": 0}
    
    # Nombre de lignes et premiers échantillons
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = 0
            first_lines = []
            delimiters = {',': 0, ';': 0, '\t': 0, '|': 0}
            
            for i, line in enumerate(f):
                lines += 1
                if i < 5:  # Capturer les 5 premières lignes
                    first_lines.append(line.strip())
                    # Compter les délimiteurs potentiels
                    for d in delimiters:
                        delimiters[d] += line.count(d)
                
                if i == 0:  # Analyser l'en-tête
                    for d in delimiters:
                        if d in line:
                            # Nombre de champs avec ce délimiteur
                            delimiters[d] = len(line.split(d))
            
            diagnostics["num_lines"] = lines
            diagnostics["first_lines"] = first_lines
            diagnostics["delimiters_found"] = delimiters
            
            # Déterminer le délimiteur probable
            max_delimiter = max(delimiters.items(), key=lambda x: x[1])
            diagnostics["likely_delimiter"] = max_delimiter[0]
            
            diagnostics["status"] = "ok"
            
    except Exception as e:
        diagnostics["status"] = "error"
        diagnostics["message"] = f"Erreur lors de l'analyse du fichier: {str(e)}"
    
    return diagnostics

def get_ingestion_data(run_id: Optional[str] = None) -> tuple:
    """
    Récupère les données ingérées depuis MLflow.
    
    Args:
        run_id (Optional[str]): ID du run MLflow contenant les données.
                              Si None, utilise le dernier run réussi.
    
    Returns:
        tuple: (data_path, effective_run_id, run_source)
               - data_path: Chemin vers le fichier de données traitées
               - effective_run_id: ID du run effectivement utilisé
               - run_source: Source du run (spécifié ou auto-détecté)
    """
    # Génération d'un ID unique pour cette opération
    op_id = str(uuid.uuid4())[:8]
    logger.info(f"[{op_id}] Récupération des données d'ingestion - Run ID: {run_id or 'Auto (dernier run)'}")
    
    
    client = get_mlflow_client()
    
    effective_run_id = run_id
    run_source = "spécifié"
    
    if effective_run_id is None:
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
            
        effective_run_id = runs[0].info.run_id
        run_source = "auto-détecté (latest)"
        
        # Récupérer des métadonnées supplémentaires sur ce run
        run = client.get_run(effective_run_id)
        run_date = datetime.fromtimestamp(run.info.start_time/1000).strftime('%Y-%m-%d %H:%M:%S')
        run_user = run.data.tags.get('mlflow.user', 'inconnu')
        logger.info(f"[{op_id}] Dernier run trouvé: {effective_run_id} (créé le {run_date} par {run_user})")
    
    # Téléchargement des artifacts du run spécifié
    logger.info(f"[{op_id}] Téléchargement des artifacts du run {effective_run_id} ({run_source})")
    start_time = time.time()
    
    # Récupérer d'abord la liste de tous les artifacts du run
    logger.info(f"[{op_id}] Listing des artifacts disponibles dans le run {effective_run_id}")
    try:
        artifacts = client.list_artifacts(effective_run_id)
        logger.info(f"[{op_id}] Artifacts disponibles dans le run: {[a.path for a in artifacts]}")
        
        # Vérifier si data_processed est présent dans les artefacts
        data_processed_exists = any(a.path == "data_processed" or a.path.startswith("data_processed/") for a in artifacts)
        logger.info(f"[{op_id}] data_processed {'existe' if data_processed_exists else 'n existe pas'} dans les artifacts")
    except Exception as e:
        logger.warning(f"[{op_id}] Erreur lors du listing des artifacts: {str(e)}")
        data_processed_exists = False
    
    try:
        # Essayer d'abord le chemin direct comme prévu dans data_ingestion.py
        logger.info(f"[{op_id}] Tentative de téléchargement depuis 'data_processed'...")
        artifacts_dir = client.download_artifacts(effective_run_id, "data_processed")
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
            artifacts_dir = client.download_artifacts(effective_run_id, "")
            logger.info(f"[{op_id}] Tous les artifacts téléchargés dans: {artifacts_dir}")
            
            # Rechercher récursivement tous les fichiers CSV
            csv_files = []
            for root, dirs, files in os.walk(artifacts_dir):
                logger.debug(f"[{op_id}] Parcours de {root}: {files}")
                csv_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
            
            logger.info(f"[{op_id}] Fichiers CSV trouvés après recherche récursive: {csv_files}")
        except Exception as second_e:
            logger.error(f"[{op_id}] Échec également lors du téléchargement depuis la racine: {str(second_e)}")
            raise ValueError(f"Impossible de récupérer les artifacts du run {effective_run_id}: {str(e)} puis {str(second_e)}")
            
    logger.info(f"[{op_id}] Artifacts téléchargés en {time.time() - start_time:.3f}s - Chemin: {artifacts_dir}")
    logger.info(f"[{op_id}] Fichiers CSV trouvés: {csv_files}")
    
    if not csv_files:
        logger.error(f"[{op_id}] Aucun fichier CSV trouvé dans les artifacts du run {run_id}")
        raise ValueError(f"Aucun fichier CSV trouvé dans les artifacts du run {run_id}")
    
    # Utiliser directement le chemin complet du premier CSV trouvé
    data_path = csv_files[0]
    logger.info(f"[{op_id}] Fichier de données trouvé: {data_path}")
    
    return data_path, effective_run_id, run_source

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
    # Journalisation des paramètres d'entrée
    train_id = str(uuid.uuid4())[:8]
    logger.info(f"[{train_id}] Démarrage de l'entraînement avec paramètres: run_id={run_id or 'latest'}, model_name={model_name or MODEL_NAME}, "
                f"base_model_name={base_model_name or MODEL_NAME}, base_model_version={base_model_version or 'latest'}")
                
    # Vérification de l'environnement
    logger.info(f"[{train_id}] Vérification de l'environnement: Python version={sys.version}, Pandas version={pd.__version__}")
    
    # Vérification des chemins
    logger.info(f"[{train_id}] Chemin de vectoriseur attendu: {VECTORIZER_PATH} (existe: {os.path.exists(VECTORIZER_PATH)})")
    
    # Vérification des variables d'environnement MLflow
    logger.info(f"[{train_id}] MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI}")
    
    # Vérification de tensorflow

    # Configuration de MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(TRAINING_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        try:
            # Chargement et préparation des données
            data_path, effective_run_id, run_source = get_ingestion_data(run_id)
            
            # Log des informations sur le run d'ingestion utilisé
            logger.info(f"[{train_id}] Utilisation du run d'ingestion: {effective_run_id} ({run_source})")
            
            # Vérifier si le fichier existe et n'est pas vide
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Le fichier de données {data_path} n'existe pas")
            
            file_size = os.path.getsize(data_path)
            if file_size == 0:
                raise ValueError(f"Le fichier de données {data_path} est vide")
            
            # Diagnostiquer le fichier CSV avant de le lire
            logger.info(f"Diagnostic du fichier CSV: {data_path}")
            diagnostics = diagnose_csv_file(data_path)
            
            if diagnostics["status"] == "error":
                logger.error(f"Diagnostic du CSV a échoué: {diagnostics.get('message', 'Raison inconnue')}")
            else:
                logger.info(f"Fichier CSV diagnostiqué: {diagnostics['num_lines']} lignes, " 
                           f"délimiteur probable: '{diagnostics['likely_delimiter']}', "
                           f"taille: {diagnostics['file_size_mb']:.2f} MB")
                
                # Journaliser les premières lignes pour aider au diagnostic
                logger.info("Premières lignes du fichier:")
                for i, line in enumerate(diagnostics["first_lines"]):
                    logger.info(f"Ligne {i}: {line[:150]}..." if len(line) > 150 else f"Ligne {i}: {line}")
                    
            # Déterminer les paramètres de lecture CSV en fonction du diagnostic
            read_params = {
                "low_memory": False,
                "escapechar": '\\',
                "on_bad_lines": 'skip'
            }
            
            # Ajouter le délimiteur si détecté
            if diagnostics.get("likely_delimiter") and diagnostics["likely_delimiter"] != ',':
                read_params["sep"] = diagnostics["likely_delimiter"]
                logger.info(f"Utilisation du délimiteur personnalisé: '{read_params['sep']}'")
            
            # Essayer de lire avec des options robustes pour traiter les fichiers problématiques
            logger.info(f"Tentative de lecture du CSV avec les paramètres: {read_params}")
            data = pd.read_csv(data_path, **read_params)
            
            if not all(col in data.columns for col in REQUIRED_COLUMNS):
                raise ValueError(f"Le fichier doit contenir les colonnes {REQUIRED_COLUMNS}")
            
            # Préparation des features et labels
            # Vérifier la présence de valeurs nulles dans les colonnes clés
            null_notes = data['Note'].isnull().sum()
            null_avis = data['Avis'].isnull().sum()
            
            if null_notes > 0 or null_avis > 0:
                logger.warning(f"Détection de valeurs nulles: {null_notes} dans 'Note', {null_avis} dans 'Avis'")
                # Filtrer les lignes avec des valeurs nulles
                data = data.dropna(subset=['Note', 'Avis'])
                logger.info(f"Après suppression des valeurs nulles: {len(data)} lignes")
                
                if len(data) == 0:
                    raise ValueError("Après suppression des valeurs nulles, aucune donnée ne reste pour l'entraînement")
            
            # Convertir les valeurs de Note en numérique si ce n'est pas déjà le cas
            if data['Note'].dtype == 'object':
                try:
                    data['Note'] = pd.to_numeric(data['Note'], errors='coerce')
                    # Supprimer les lignes où la conversion a échoué
                    data = data.dropna(subset=['Note'])
                    logger.info(f"Conversion de 'Note' en numérique: {len(data)} lignes restantes")
                except Exception as e:
                    logger.error(f"Erreur lors de la conversion des notes: {str(e)}")
                    raise ValueError(f"Impossible de convertir la colonne 'Note' en valeurs numériques: {str(e)}")
            
            # S'assurer que 'Avis' est de type string
            data['Avis'] = data['Avis'].astype(str)
            
            # Générer les labels
            y = (data['Note'] > POSITIVE_REVIEW_THRESHOLD).astype(int)
            
            # Charger le vectoriseur
            try:
                with open(VECTORIZER_PATH, 'rb') as f:
                    vectorizer = pickle.load(f)
            except Exception as e:
                logger.error(f"Erreur lors du chargement du vectoriseur: {str(e)}")
                raise ValueError(f"Impossible de charger le vectoriseur depuis {VECTORIZER_PATH}: {str(e)}")

            # Déterminer la colonne à utiliser pour les features: 'Mots_importants' si disponible, sinon 'Avis'
            feature_column = 'Mots_importants' if 'Mots_importants' in data.columns else 'Avis'
            logger.info(f"Utilisation de la colonne '{feature_column}' comme données d'entrée pour l'entraînement")
            
            # S'assurer que la colonne utilisée est de type string
            data[feature_column] = data[feature_column].astype(str)
                
            # Transformer le texte en features
            try:
                X = vectorizer.transform(data[feature_column])
            except Exception as e:
                logger.error(f"Erreur lors de la vectorisation des textes de la colonne '{feature_column}': {str(e)}")
                raise ValueError(f"Impossible de vectoriser les textes: {str(e)}")
            
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
            mlflow.log_param("ingestion_run_id", effective_run_id)
            mlflow.log_param("ingestion_run_id_source", run_source)  
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("feature_column", feature_column)
            
            # Log des références du modèle d'origine
            effective_base_model = base_model_name or MODEL_NAME
            effective_base_version = base_model_version
            
            # Si la version du modèle de base n'est pas spécifiée, journaliser la version réelle utilisée
            if not effective_base_version:
                try:
                    # Récupérer la dernière version du modèle de base
                    client = MlflowClient()
                    latest = get_latest_registered_version(client, effective_base_model)
                    effective_base_version = latest.version if latest else "non trouvé"
                    base_version_source = "auto-détecté (latest)"
                except Exception as e:
                    logger.warning(f"[{train_id}] Impossible de récupérer la dernière version du modèle de base: {str(e)}")
                    effective_base_version = "unknown"
                    base_version_source = "erreur de détection"
            else:
                base_version_source = "spécifié"
                
            mlflow.log_param("base_model_name", effective_base_model)
            mlflow.log_param("base_model_version", effective_base_version)
            mlflow.log_param("base_model_version_source", base_version_source)
            
            # Log des métadonnées du dataset
            try:
                dataset_info = {
                    "dataset_source": data_path,
                    "dataset_size": len(data),
                    "dataset_features": list(data.columns),
                    "positive_samples": int(sum(y)),
                    "negative_samples": int(len(y) - sum(y)),
                    "feature_column_used": feature_column,  # Indique quelle colonne a été utilisée pour les features
                    "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                mlflow.log_dict(dataset_info, "dataset_info.json")
                logger.info(f"Métadonnées du dataset loggées: {len(data)} échantillons, colonne utilisée: {feature_column}")
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
                "model_version": latest_version.version,
                # Informations de traçabilité
                "ingestion_run_id": effective_run_id,
                "ingestion_run_id_source": run_source,
                "base_model_name": effective_base_model,
                "base_model_version": effective_base_version,
                "base_model_version_source": base_version_source
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