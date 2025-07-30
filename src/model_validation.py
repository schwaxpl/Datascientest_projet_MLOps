"""
Module de validation du modèle.
Permet de valider un modèle avant sa mise en production.
"""

import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from typing import Dict, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import os
import time
import uuid
from datetime import datetime
from src.logger_config import get_logger

# Configuration du logger spécifique au module de validation
logger = get_logger('model_validation')

from src.config import (
    MODEL_NAME,
    VECTORIZER_PATH,
    MLFLOW_TRACKING_URI,
    VALIDATION_THRESHOLD
)

from src.utils import load_model_from_registry, get_latest_registered_version

def get_models_to_validate(client: MlflowClient, model_name: Optional[str] = None) -> List[Dict]:
    """
    Récupère la liste des modèles en attente de validation.
    
    Args:
        client: MLflow client
        model_name: Nom spécifique du modèle à rechercher (optionnel)
        
    Returns:
        Liste des modèles à valider (avec leurs métadonnées)
    """
    validation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{validation_id}] Recherche des modèles à valider")
    
    if model_name:
        model_names = [model_name]
        logger.info(f"[{validation_id}] Recherche limitée au modèle: {model_name}")
    else:
        # Récupérer tous les modèles du registre
        registered_models = client.search_registered_models()
        model_names = [rm.name for rm in registered_models]
        logger.info(f"[{validation_id}] Recherche sur {len(model_names)} modèles dans le registre")
    
    models_to_validate = []
    
    for name in model_names:
        # Recherche des versions en staging avec le tag "status"="à valider"
        versions = client.search_model_versions(f"name='{name}'")
        for version in versions:
            if version.current_stage == "Staging":
                # Récupérer les tags pour vérifier si le modèle est en attente de validation
                tags = client.get_model_version_tags(name, version.version)
                if "status" in tags and tags["status"] == "à valider":
                    logger.info(f"[{validation_id}] Modèle trouvé pour validation: {name} version {version.version}")
                    models_to_validate.append({
                        "name": name,
                        "version": version.version,
                        "run_id": version.run_id,
                        "timestamp": version.creation_timestamp
                    })
    
    logger.info(f"[{validation_id}] {len(models_to_validate)} modèles trouvés en attente de validation")
    return models_to_validate

def prepare_validation_data(client: MlflowClient, model_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prépare les données de validation pour un modèle spécifique.
    
    Args:
        client: MLflow client
        model_info: Informations sur le modèle à valider
        
    Returns:
        Tuple (X_validation, y_validation) pour évaluation
    """
    validation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{validation_id}] Préparation des données de validation pour {model_info['name']} v{model_info['version']}")
    
    # Récupération du run qui a créé ce modèle
    run = client.get_run(model_info["run_id"])
    
    # Récupération du run_id d'ingestion utilisé pour l'entraînement
    ingestion_run_id = run.data.params.get("ingestion_run_id")
    if not ingestion_run_id or ingestion_run_id == "latest":
        logger.warning(f"[{validation_id}] Pas de run d'ingestion spécifique, recherche du dernier jeu de validation")
        # Trouver un jeu de données de validation
        experiment = mlflow.get_experiment_by_name("data_ingestion_api")
        if not experiment:
            logger.error(f"[{validation_id}] Aucune expérience d'ingestion trouvée")
            raise ValueError("Aucune expérience d'ingestion de données trouvée")
        
        validation_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.dataset_type = 'jdd validation'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not validation_runs:
            logger.warning(f"[{validation_id}] Aucun jeu de validation spécifique trouvé. Veuillez uploader un jeu de validation via '/upload/validation'")
            logger.warning(f"[{validation_id}] Utilisation du dernier jeu d'entraînement comme fallback (non recommandé)")
            validation_runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.dataset_type = 'jdd entrainement' AND status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
        if not validation_runs:
            logger.error(f"[{validation_id}] Aucun jeu de données disponible pour validation")
            raise ValueError("Aucun jeu de données disponible pour validation. Veuillez uploader un jeu de validation via '/upload/validation'")
            
        ingestion_run_id = validation_runs[0].info.run_id
    
    logger.info(f"[{validation_id}] Utilisation du jeu de données du run: {ingestion_run_id}")
    
    # Téléchargement des artifacts du run d'ingestion
    start_time = time.time()
    
    # Récupérer d'abord la liste de tous les artifacts du run
    logger.info(f"[{validation_id}] Listing des artifacts disponibles dans le run {ingestion_run_id}")
    try:
        artifacts = client.list_artifacts(ingestion_run_id)
        logger.info(f"[{validation_id}] Artifacts disponibles: {[a.path for a in artifacts]}")
    except Exception as e:
        logger.warning(f"[{validation_id}] Erreur lors du listing des artifacts: {str(e)}")
    
    try:
        # Essayer d'abord le chemin direct
        artifacts_dir = client.download_artifacts(ingestion_run_id, "data_processed")
        logger.info(f"[{validation_id}] Artifacts téléchargés en {time.time() - start_time:.3f}s - Chemin: {artifacts_dir}")
        
        # Trouver le fichier CSV
        csv_files = [f for f in os.listdir(artifacts_dir) if f.endswith('.csv')]
    except Exception as e:
        logger.warning(f"[{validation_id}] Erreur lors du téléchargement depuis 'data_processed': {str(e)}")
        logger.info(f"[{validation_id}] Tentative de téléchargement depuis la racine...")
        
        try:
            # Plan B: télécharger tous les artefacts et chercher les CSVs
            artifacts_dir = client.download_artifacts(ingestion_run_id, "")
            logger.info(f"[{validation_id}] Tous les artifacts téléchargés")
            
            # Rechercher récursivement tous les fichiers CSV
            csv_files = []
            for root, dirs, files in os.walk(artifacts_dir):
                csv_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
            
            # Convertir les chemins complets en noms de fichiers relatifs pour le log
            csv_file_names = [os.path.basename(f) for f in csv_files]
            logger.info(f"[{validation_id}] Fichiers CSV trouvés: {csv_file_names}")
        except Exception as second_e:
            logger.error(f"[{validation_id}] Échec également lors du téléchargement depuis la racine: {str(second_e)}")
            raise ValueError(f"Impossible de récupérer les artifacts du run {ingestion_run_id}")
    
    if not csv_files:
        logger.error(f"[{validation_id}] Aucun fichier CSV trouvé dans les artifacts du run {ingestion_run_id}")
        raise ValueError(f"Aucun fichier CSV trouvé dans les artifacts du run {ingestion_run_id}")
    
    # Charger les données (utiliser le premier fichier CSV si plusieurs sont trouvés)
    data_path = csv_files[0] if os.path.isabs(csv_files[0]) else os.path.join(artifacts_dir, csv_files[0])
    logger.info(f"[{validation_id}] Fichier de données trouvé: {data_path}")
    
    from src.config import REQUIRED_COLUMNS, POSITIVE_REVIEW_THRESHOLD
    
    # Chargement des données
    data = pd.read_csv(data_path)
    
    # Vérifier les colonnes requises
    if not all(col in data.columns for col in REQUIRED_COLUMNS):
        logger.error(f"[{validation_id}] Le fichier doit contenir les colonnes {REQUIRED_COLUMNS}")
        raise ValueError(f"Le fichier doit contenir les colonnes {REQUIRED_COLUMNS}")
    
    # Préparation des features et labels
    y = (data['Note'] > POSITIVE_REVIEW_THRESHOLD).astype(int)
    
    # Charger le vectorizer
    with open(VECTORIZER_PATH, 'rb') as f:
        import pickle
        vectorizer = pickle.load(f)
    
    # Transformer le texte
    X = vectorizer.transform(data['Avis'])
    
    logger.info(f"[{validation_id}] Données préparées: {X.shape[0]} échantillons")
    
    return X, y

def validate_model(model_name: Optional[str] = None, model_version: Optional[str] = None, 
                   approve: bool = False, threshold: Optional[float] = None) -> Dict:
    """
    Valide un modèle spécifique ou tous les modèles en attente de validation.
    
    Args:
        model_name: Nom du modèle à valider (optionnel)
        model_version: Version du modèle à valider (optionnel)
        approve: Si True, approuve automatiquement les modèles qui dépassent le seuil
        threshold: Seuil d'accuracy pour l'approbation automatique (utilise VALIDATION_THRESHOLD par défaut)
        
    Returns:
        Dict: Résultats de la validation
    """
    validation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{validation_id}] Démarrage du processus de validation")
    
    # Configuration de MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    from src.utils import get_mlflow_client
    client = get_mlflow_client()
    
    # Utiliser le seuil de configuration si non spécifié
    if threshold is None:
        threshold = VALIDATION_THRESHOLD
    logger.info(f"[{validation_id}] Seuil d'approbation: {threshold}")
    
    # Si un modèle et une version spécifiques sont demandés
    if model_name and model_version:
        models_to_validate = [{
            "name": model_name,
            "version": model_version,
            "run_id": client.get_model_version(model_name, model_version).run_id,
            "timestamp": client.get_model_version(model_name, model_version).creation_timestamp
        }]
    else:
        models_to_validate = get_models_to_validate(client, model_name)
    
    if not models_to_validate:
        logger.info(f"[{validation_id}] Aucun modèle à valider trouvé")
        return {"status": "info", "message": "Aucun modèle à valider trouvé"}
    
    validation_results = []
    
    for model_info in models_to_validate:
        logger.info(f"[{validation_id}] Validation du modèle {model_info['name']} version {model_info['version']}")
        
        try:
            # Charger le modèle
            model = load_model_from_registry(model_info["name"], version=model_info["version"])
            
            # Préparer les données de validation
            X_val, y_val = prepare_validation_data(client, model_info)
            
            # Vérifier si nous utilisons un jeu de validation dédié
            validation_id_for_run = str(uuid.uuid4())[:8]
            run = client.get_run(model_info["run_id"])
            ingestion_run_id = run.data.params.get("ingestion_run_id")
            if ingestion_run_id and ingestion_run_id != "latest":
                ingestion_run = client.get_run(ingestion_run_id)
                using_dedicated_validation = ingestion_run.data.tags.get("dataset_type") == "jdd validation"
            else:
                using_dedicated_validation = False
            
            # Évaluer le modèle
            start_time = time.time()
            y_pred = np.argmax(model.predict(X_val.toarray()), axis=1)
            eval_time = time.time() - start_time
            
            # Calculer les métriques
            accuracy = accuracy_score(y_val, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
            
            logger.info(f"[{validation_id}] Résultats: Accuracy={accuracy:.4f}, F1={f1:.4f}, temps={eval_time:.3f}s")
            
            # Rechercher la version en production du même modèle pour comparaison
            production_metrics = None
            try:
                # Recherche des versions en production pour ce modèle
                prod_versions = [v for v in client.search_model_versions(f"name='{model_info['name']}'") 
                                if v.current_stage == "Production"]
                
                if prod_versions:
                    logger.info(f"[{validation_id}] Version en production trouvée pour {model_info['name']}: version {prod_versions[0].version}")
                    
                    # Charger le modèle de production pour comparaison
                    production_model = load_model_from_registry(model_info["name"], version=prod_versions[0].version)
                    
                    # Évaluer le modèle de production sur les mêmes données
                    prod_start_time = time.time()
                    y_prod_pred = np.argmax(production_model.predict(X_val.toarray()), axis=1)
                    prod_eval_time = time.time() - prod_start_time
                    
                    # Calculer les métriques
                    prod_accuracy = accuracy_score(y_val, y_prod_pred)
                    prod_precision, prod_recall, prod_f1, _ = precision_recall_fscore_support(
                        y_val, y_prod_pred, average='weighted'
                    )
                    
                    production_metrics = {
                        "version": prod_versions[0].version,
                        "accuracy": float(prod_accuracy),
                        "precision": float(prod_precision),
                        "recall": float(prod_recall),
                        "f1_score": float(prod_f1),
                        "evaluation_time": float(prod_eval_time)
                    }
                    
                    # Comparaison des performances
                    accuracy_diff = accuracy - prod_accuracy
                    f1_diff = f1 - prod_f1
                    
                    logger.info(f"[{validation_id}] Comparaison avec la version en production:")
                    logger.info(f"[{validation_id}]   - Nouveau modèle: Accuracy={accuracy:.4f}, F1={f1:.4f}")
                    logger.info(f"[{validation_id}]   - Production: Accuracy={prod_accuracy:.4f}, F1={prod_f1:.4f}")
                    logger.info(f"[{validation_id}]   - Différence: Accuracy={accuracy_diff:.4f}, F1={f1_diff:.4f}")
                    
                    # Afficher un avertissement si le modèle candidat est moins performant
                    if accuracy < prod_accuracy:
                        logger.warning(f"[{validation_id}] AVERTISSEMENT: Le nouveau modèle a une précision inférieure au modèle en production!")
                else:
                    logger.info(f"[{validation_id}] Aucune version en production trouvée pour {model_info['name']}")
            except Exception as e:
                logger.warning(f"[{validation_id}] Erreur lors de la comparaison avec le modèle en production: {str(e)}")
                # Ne pas bloquer la validation en cas d'erreur de comparaison
            
            # Décision de validation avec deux critères:
            # 1. Le modèle doit dépasser le seuil absolu
            # 2. S'il y a un modèle en production, le nouveau modèle ne doit pas être significativement moins performant
            meets_threshold = accuracy >= threshold
            
            # Définir une marge de tolérance pour la régression (0.01 = 1%)
            regression_tolerance = 0.01
            
            if production_metrics:
                # Vérifier si le nouveau modèle n'est pas significativement moins bon que le modèle en production
                not_significant_regression = (accuracy >= production_metrics["accuracy"] - regression_tolerance)
                
                # Pour l'approbation finale, les deux critères doivent être satisfaits
                approved = meets_threshold and not_significant_regression
                
                if not not_significant_regression and meets_threshold:
                    logger.warning(f"[{validation_id}] Le modèle satisfait le seuil mais représente une régression par rapport à la production")
            else:
                # S'il n'y a pas de modèle en production, utiliser uniquement le seuil
                approved = meets_threshold
            
            result = {
                "model_name": model_info["name"],
                "model_version": model_info["version"],
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "evaluation_time": float(eval_time),
                "samples_count": int(X_val.shape[0]),
                "threshold": float(threshold),
                "approved": approved,
                "using_dedicated_validation_dataset": using_dedicated_validation,
                "validation_data_quality": "high" if using_dedicated_validation else "standard",
                "timestamp": datetime.now().isoformat()
            }
            
            # Ajouter les métriques de comparaison si disponibles
            if production_metrics:
                # Calcul de la régression éventuelle
                accuracy_diff = accuracy - production_metrics["accuracy"]
                is_regression = accuracy_diff < -regression_tolerance
                is_improvement = accuracy_diff > 0
                
                result["production_comparison"] = {
                    "production_version": production_metrics["version"],
                    "production_accuracy": production_metrics["accuracy"],
                    "production_precision": production_metrics["precision"],
                    "production_recall": production_metrics["recall"],
                    "production_f1_score": production_metrics["f1_score"],
                    "accuracy_difference": float(accuracy_diff),
                    "f1_difference": float(f1 - production_metrics["f1_score"]),
                    "regression_tolerance": float(regression_tolerance),
                    "is_significant_regression": bool(is_regression),
                    "is_improvement": bool(is_improvement),
                    "validation_criteria": {
                        "meets_absolute_threshold": bool(meets_threshold),
                        "not_significant_regression": bool(not is_regression),
                        "both_criteria_met": bool(approved)
                    }
                }
            
            # Log des résultats de validation dans MLflow
            with mlflow.start_run(run_name=f"validation_{model_info['name']}_v{model_info['version']}"):
                mlflow.log_params({
                    "validation_id": validation_id,
                    "model_name": model_info["name"],
                    "model_version": model_info["version"],
                    "threshold": threshold,
                    "using_dedicated_validation_dataset": using_dedicated_validation,
                })
                
                metrics_to_log = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "evaluation_time": eval_time,
                    "samples_count": X_val.shape[0],
                }
                
                # Ajouter les métriques comparatives si disponibles
                if production_metrics:
                    accuracy_diff = accuracy - production_metrics["accuracy"]
                    is_regression = accuracy_diff < -regression_tolerance
                    
                    metrics_to_log.update({
                        "production_accuracy": production_metrics["accuracy"],
                        "production_f1_score": production_metrics["f1_score"],
                        "accuracy_difference": accuracy_diff,
                        "f1_difference": f1 - production_metrics["f1_score"],
                        "meets_absolute_threshold": int(meets_threshold),
                        "is_significant_regression": int(is_regression),
                    })
                    mlflow.set_tag("compared_to_production", "true")
                    mlflow.set_tag("production_version", production_metrics["version"])
                    mlflow.set_tag("comparison_result", 
                                  "improvement" if accuracy_diff > 0 else 
                                  "regression" if is_regression else "neutral")
                
                mlflow.log_metrics(metrics_to_log)
                
                # Créer un tag pour lier la validation au run du modèle
                mlflow.set_tag("model_run_id", model_info["run_id"])
                mlflow.set_tag("validation_type", "automatic")
                
                # Si le modèle est approuvé et que l'approbation automatique est activée
                if approved and approve:
                    logger.info(f"[{validation_id}] Le modèle {model_info['name']} v{model_info['version']} "
                               f"a passé la validation (accuracy: {accuracy:.4f} >= {threshold})")
                    
                    # Transition vers Production
                    client.transition_model_version_stage(
                        name=model_info["name"],
                        version=model_info["version"],
                        stage="Production"
                    )
                    
                    # Mise à jour des tags
                    client.set_model_version_tag(
                        name=model_info["name"],
                        version=model_info["version"],
                        key="status",
                        value="production"
                    )
                    
                    client.set_model_version_tag(
                        name=model_info["name"],
                        version=model_info["version"],
                        key="validation_accuracy",
                        value=str(accuracy)
                    )
                    
                    client.set_model_version_tag(
                        name=model_info["name"],
                        version=model_info["version"],
                        key="validation_date",
                        value=datetime.now().isoformat()
                    )
                    
                    mlflow.set_tag("validation_result", "approved")
                    result["action_taken"] = "promoted_to_production"
                    
                    logger.info(f"[{validation_id}] Modèle {model_info['name']} v{model_info['version']} "
                               f"promu en production")
                else:
                    if approved:
                        logger.info(f"[{validation_id}] Le modèle {model_info['name']} v{model_info['version']} "
                                  f"a passé la validation mais l'approbation automatique est désactivée")
                        mlflow.set_tag("validation_result", "passed_no_action")
                        result["action_taken"] = "none"
                    else:
                        logger.info(f"[{validation_id}] Le modèle {model_info['name']} v{model_info['version']} "
                                  f"n'a pas passé la validation (accuracy: {accuracy:.4f} < {threshold})")
                        mlflow.set_tag("validation_result", "failed")
                        result["action_taken"] = "none"
            
            validation_results.append(result)
            
        except Exception as e:
            logger.error(f"[{validation_id}] Erreur lors de la validation du modèle {model_info['name']} "
                       f"v{model_info['version']}: {str(e)}", exc_info=True)
            validation_results.append({
                "model_name": model_info["name"],
                "model_version": model_info["version"],
                "error": str(e),
                "status": "error"
            })
    
    logger.info(f"[{validation_id}] Validation terminée pour {len(models_to_validate)} modèles")
    
    return {
        "status": "success",
        "validation_id": validation_id,
        "models_validated": len(validation_results),
        "results": validation_results
    }

def validate_and_promote_model(model_name: str, model_version: str) -> Dict:
    """
    Valide un modèle spécifique et le promeut en production s'il est validé.
    
    Args:
        model_name: Nom du modèle à valider
        model_version: Version du modèle à valider
        
    Returns:
        Dict: Résultat de la validation et de la promotion
    """
    # Effectuer la validation avec approbation automatique
    validation_result = validate_model(
        model_name=model_name,
        model_version=model_version,
        approve=True
    )
    
    return validation_result

if __name__ == "__main__":
    # Si exécuté directement, valider tous les modèles en attente sans promotion automatique
    result = validate_model()
    print(f"Résultats de la validation: {len(result['results'])} modèles validés")
    for model_result in result['results']:
        status = "✅ APPROUVÉ" if model_result.get('approved', False) else "❌ REFUSÉ"
        print(f"{status}: {model_result['model_name']} v{model_result['model_version']}, "
              f"Accuracy: {model_result.get('accuracy', 'N/A')}")
