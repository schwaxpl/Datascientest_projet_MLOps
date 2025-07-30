"""
API REST FastAPI pour le service de prédiction.
"""

# Imports standard de Python
import os
import json
import time
import uuid
import asyncio
import pickle
import tempfile
import functools
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Any, Callable, Union

# Imports de bibliothèques externes
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
import mlflow.tensorflow
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Query, Request, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

# Imports des modules locaux
from src.predict import PredictionService
from src.data_ingestion import DataIngestionPipeline
from src.logger_config import init_logging
from src.utils import (
    get_mlflow_client, 
    get_latest_model_version,
    load_model_from_registry, 
    get_vectorizer_from_run
)
from src.config import (
    MODEL_NAME,
    MODEL_PATH,
    VECTORIZER_PATH,
    MLFLOW_TRACKING_URI
)

# Import des modules de validation et d'entraînement (utilisés plus tard)
from src.model_validation import validate_model, validate_and_promote_model
from src.train import train_model as train_model_function

# Initialisation du système de logging
loggers = init_logging(api=True)
logger = loggers['api']

# Fonction de décorateur pour logger les appels d'API
def log_endpoint(func: Callable):
    @functools.wraps(func)
    async def wrapper(*args, request = None, **kwargs):
        # Génération d'un identifiant unique pour la requête
        request_id = str(uuid.uuid4())[:8]
        
        # Extraction des données de la requête pour le logging
        # Vérifier si l'objet request a les attributs attendus d'une requête FastAPI
        has_client = hasattr(request, 'client') and request.client is not None
        has_url = hasattr(request, 'url') and request.url is not None
        has_method = hasattr(request, 'method')
        
        client_host = request.client.host if has_client else "unknown"
        endpoint = request.url.path if has_url else func.__name__
        method = request.method if has_method else "UNKNOWN"
        
        # Log de début de requête (format standard)
        logger.info(f"[{request_id}] Début de traitement: {endpoint} - Client: {client_host}")
        
        # Mesure du temps d'exécution
        start_time = time.time()
        
        try:
            # Exécution de la fonction d'origine
            response = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Calcul du temps d'exécution
            execution_time = time.time() - start_time
            
            # Log de fin de requête réussie (avec attributs HTTP)
            if request:
                status_code = getattr(response, "status_code", 200)
                logger.info(
                    f"Endpoint traité - {request_id}",
                    extra={
                        'method': method,
                        'url': endpoint,
                        'status_code': status_code,
                        'response_time': execution_time,
                        'client_ip': client_host
                    }
                )
            else:
                # Format standard si pas de requête
                logger.info(f"[{request_id}] Traitement réussi: {func.__name__} - Temps: {execution_time:.3f}s")
            
            return response
            
        except Exception as e:
            # Calcul du temps d'exécution en cas d'erreur
            execution_time = time.time() - start_time
            
            # Log de l'erreur (format standard)
            logger.error(f"[{request_id}] Erreur lors du traitement de {endpoint} - {str(e)} - Temps: {execution_time:.3f}s", exc_info=True)
            
            # Relance de l'exception pour que FastAPI puisse la gérer
            raise
            
    return wrapper

# Création d'une classe middleware pour le logging
class LoggingMiddleware:
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
            
        # Création d'un ID de requête unique
        request_id = str(uuid.uuid4())[:8]
        
        # Extraction des informations de la requête
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "UNKNOWN")
        client = scope.get("client", ("UNKNOWN", 0))
        client_ip = client[0] if client else "UNKNOWN"
        
        # Log de début de requête (format standard pour éviter l'utilisation des attributs spéciaux)
        logger.info(f"[{request_id}] Requête reçue: {method} {path} - Client: {client_ip}")
        
        # Mesure du temps d'exécution
        start_time = time.time()
        
        # Traitement de la requête
        await self.app(scope, receive, send)
        
        # Calcul du temps d'exécution
        execution_time = time.time() - start_time
        
        # Log de fin de requête avec les attributs HTTP pour le formateur spécial
        logger.info(
            f"Requête terminée - {request_id}",
            extra={
                'method': method,
                'url': path,
                'status_code': "200",  # Nous n'avons pas accès au code de statut ici
                'response_time': execution_time,
                'client_ip': client_ip
            }
        )
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Événement exécuté au démarrage de l'application"""
    logger.info("Initialisation de l'application...")
    init_mlflow_model()
    logger.info("Initialisation terminée")
    yield
    logger.info("Arrêt de l'application")

app = FastAPI(
    title="Datascientest MLOps Analyse de Sentiments",
    description="""
    # API pour l'analyse de sentiments des avis clients
    
    Cette API permet de :
    * **Prédire** le sentiment (positif/négatif) d'un texte
    * **Charger** et traiter des fichiers CSV d'avis clients
    * **Entraîner** de nouveaux modèles à partir des données
    
    ## 🚀 Guide d'utilisation rapide:
    
    1. Pour la **prédiction** de sentiment:
       - Endpoint JSON standard: `/predict`
       - Endpoint formulaire: `/predict/form` ✅ (recommandé pour les tests, pré-rempli avec exemple)
    
    2. Pour **l'entraînement** du modèle:
       - Endpoint JSON standard: `/train`
       - Endpoint formulaire: `/train/form` ✅ (recommandé pour les tests, tous les champs optionnels)
    
    3. Pour **l'upload** de données:
       - Endpoint: `/upload` (accepte un fichier CSV)
    
    Les formulaires sont pré-remplis avec des valeurs par défaut pour faciliter les tests.
    
    Développé dans le cadre du projet MLOps Datascientest.
    """,
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Général",
            "description": "Informations générales sur l'API"
        },
        {
            "name": "Prédiction",
            "description": "Endpoints pour la prédiction de sentiments (formats JSON et formulaire)"
        },
        {
            "name": "Données",
            "description": "Endpoints pour le chargement et la gestion des données"
        },
        {
            "name": "Entraînement",
            "description": "Endpoints pour l'entraînement de modèles (formats JSON et formulaire)"
        }
    ]
)

# Ajout du middleware de logging
app.add_middleware(LoggingMiddleware)
logger.info("API démarrée avec middleware de logging activé")

# Initialisation lazy du service de prédiction et du vectorizer
prediction_service = None
vectorizer = None

def load_vectorizer():
    """Charge le vectorizer depuis le fichier local"""
    global vectorizer
    if vectorizer is None:
        logger.info(f"Chargement du vectorizer depuis {VECTORIZER_PATH}")
        try:
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            logger.info("Vectorizer chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du vectorizer: {str(e)}", exc_info=True)
            raise
    else:
        logger.debug("Utilisation du vectorizer en cache")
    return vectorizer

def init_mlflow_model():
    """
    Initialise le modèle dans MLflow Model Registry si nécessaire.
    Crée une première version à partir du modèle local si aucune version n'existe.
    """
    logger.info("Initialisation du modèle MLflow")

    # Création du client MLflow configuré pour S3/MinIO
    client = get_mlflow_client()
    
    try:
        # Vérifier si le modèle existe dans le registre
        try:
            model = client.get_registered_model(MODEL_NAME)
            logger.info(f"Modèle {MODEL_NAME} trouvé dans le registre MLflow")
        except:
            logger.info(f"Création du modèle {MODEL_NAME} dans MLflow Model Registry")
            client.create_registered_model(MODEL_NAME)
            logger.info(f"Modèle {MODEL_NAME} créé avec succès")
        
        # Vérifier s'il existe des versions
        try:
            latest_version = get_latest_model_version(client, MODEL_NAME)
            logger.info(f"Version existante trouvée: {latest_version.version}")
        except ValueError:
            logger.info("Aucune version trouvée. Création de la version initiale...")
            # S'assurer que l'expérience existe avant de démarrer le run
            # Le client MLflow est déjà configuré pour S3/MinIO par get_mlflow_client
            mlflow.set_experiment("model_training")
            experiment = mlflow.get_experiment_by_name("model_training")
            artifact_uri = f"http://mlflow:5000/artifacts/{experiment.experiment_id}"
            # Créer un run MLflow pour enregistrer le modèle initial
            with mlflow.start_run(run_name="initial_model_registration") as run:
                client.set_tag(run.info.run_id, "mlflow.artifact.uri", artifact_uri)
                logger.info(f"Démarrage d'un run MLflow (ID: {run.info.run_id})")
                try:
                    # Charger le modèle directement comme un modèle Keras
                    logger.info(f"Tentative de chargement du modèle depuis {MODEL_PATH}")
                    model = load_model(MODEL_PATH)
                    logger.info("Modèle Keras chargé avec succès")
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement du modèle comme modèle Keras: {str(e)}")
                    logger.info("Tentative de chargement comme modèle pickle...")
                    with open(MODEL_PATH, 'rb') as f:
                        model = pickle.load(f)
                    logger.info("Modèle pickle chargé avec succès")
                
                # Charger le vectorizer
                logger.info(f"Chargement du vectorizer depuis {VECTORIZER_PATH}")
                with open(VECTORIZER_PATH, 'rb') as f:
                    vectorizer = pickle.load(f)
                logger.info("Vectorizer chargé avec succès")
                
                # Log du vectorizer comme artifact (MLflow se charge de le copier dans son artifact store)
                logger.info("Enregistrement du vectorizer comme artifact MLflow")
                try:
                    # Utiliser directement le chemin du vectorizer local sans créer de fichier temporaire
                    mlflow.log_artifact(VECTORIZER_PATH, "vectorizer")
                    logger.info("Vectorizer enregistré avec succès comme artifact MLflow")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'enregistrement du vectorizer comme artifact: {str(e)}")
                    logger.warning("Continuons avec le log du modèle sans l'artifact vectorizer")
                
                # Log du modèle avec MLflow, en spécifiant explicitement keras comme flavor
                logger.info("Enregistrement du modèle dans MLflow")
                
                # Fallback à la méthode standard si la première échoue
                mlflow.tensorflow.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=MODEL_NAME
                )
                logger.info(f"Modèle enregistré avec succès (méthode standard): {MODEL_NAME}")
            
                # Récupérer la dernière version créée
                latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                if latest_versions:
                    latest_version = latest_versions[0]
                    # Mettre le tag "production" pour la version initiale
                    logger.info(f"Application du tag 'production' pour la version {latest_version.version}")
                    client.transition_model_version_stage(
                        name=MODEL_NAME,
                        version=latest_version.version,
                        stage="Production"
                    )
                    logger.info(f"Version {latest_version.version} marquée comme 'Production'")
                    
            logger.info("Version initiale créée avec succès")
            
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du modèle dans MLflow: {str(e)}", exc_info=True)
        raise

def get_prediction_service(model_name: Optional[str] = None, model_version: Optional[str] = None):
    """
    Retourne une instance du service de prédiction.
    
    Args:
        model_name: Nom du modèle à utiliser. Si None, utilise MODEL_NAME de la configuration
        model_version: Version spécifique du modèle. Si None, utilise la dernière version disponible
        
    Returns:
        PredictionService: Instance du service de prédiction
    """
    global prediction_service
    
    # Si un modèle spécifique est demandé, ne pas utiliser le cache
    if model_name or model_version:
        logger.info(f"Chargement d'un modèle spécifique: {model_name or MODEL_NAME}, version: {model_version or 'latest'}")
        # Configuration du client MLflow configuré pour S3/MinIO
        
        # Chargement du modèle depuis MLflow
        client = get_mlflow_client()
        logger.info(f"Chargement du modèle depuis le registre MLflow: {model_name or MODEL_NAME}, version: {model_version or 'latest'}")
        model = load_model_from_registry(model_name or MODEL_NAME, version=model_version)
        logger.info("Modèle chargé avec succès")
        
        # Utilisation du vectorizer local
        logger.info("Chargement du vectorizer local")
        vectorizer = load_vectorizer()
        
        logger.info("Création du service de prédiction avec modèle et vectorizer")
        return PredictionService.from_artifacts(
            model=model,
            vectorizer=vectorizer
        )
        
    # Utiliser le cache si aucun modèle spécifique n'est demandé
    if prediction_service is None:
        logger.info(f"Initialisation du service de prédiction avec le modèle par défaut: {MODEL_NAME}")
        
        # Récupération de la dernière version du modèle avec le client configuré pour S3/MinIO
        client = get_mlflow_client()
        latest_version = get_latest_model_version(client, MODEL_NAME)
        logger.info(f"Chargement du modèle version: {latest_version.version}")
        
        # Chargement du modèle depuis MLflow
        model = load_model_from_registry(MODEL_NAME)
        logger.info("Modèle chargé avec succès")
        
        # Utilisation du vectorizer local
        logger.info("Chargement du vectorizer local")
        vectorizer = load_vectorizer()
        
        # Création du service avec le modèle et le vectorizer
        logger.info("Création du service de prédiction avec modèle et vectorizer")
        prediction_service = PredictionService.from_artifacts(
            model=model,
            vectorizer=vectorizer
        )
        logger.info("Service de prédiction initialisé avec succès")
    else:
        logger.info("Utilisation du service de prédiction en cache")
        
    return prediction_service

class PredictionRequest(BaseModel):
    """Modèle pour une requête de prédiction de sentiment"""
    text: str = Field(
        ..., 
        title="Texte à analyser", 
        description="Le texte de l'avis client pour lequel vous souhaitez prédire le sentiment",
        example="Ce produit est vraiment excellent, je le recommande vivement !"
    )
    model_name: Optional[str] = Field(
        None, 
        title="Nom du modèle", 
        description="Nom du modèle à utiliser pour la prédiction (optionnel)",
        example="dst_trustpilot"
    )
    model_version: Optional[str] = Field(
        None, 
        title="Version du modèle", 
        description="Version spécifique du modèle à utiliser (optionnel)",
        example="1"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Ce produit est vraiment excellent, je le recommande vivement !",
                "model_name": "dst_trustpilot",
                "model_version": "1"
            }
        }

class PredictionResponse(BaseModel):
    """Réponse de prédiction de sentiment"""
    prediction: int = Field(
        ..., 
        title="Prédiction", 
        description="0 pour sentiment négatif, 1 pour sentiment positif",
        example=1
    )
    probabilities: Dict[str, float] = Field(
        ..., 
        title="Probabilités", 
        description="Probabilités pour chaque classe (négatif et positif)",
        example={"négatif": 0.1, "positif": 0.9}
    )
    sentiment: str = Field(
        ..., 
        title="Sentiment", 
        description="Sentiment en texte: 'négatif' ou 'positif'",
        example="positif"
    )

class TrainingRequest(BaseModel):
    """Modèle pour une requête d'entraînement"""
    run_id: Optional[str] = Field(
        None, 
        title="ID du run MLflow", 
        description="ID du run MLflow contenant les données d'entraînement (optionnel)",
        example="a1b2c3d4e5f6"
    )
    model_name: Optional[str] = Field(
        None, 
        title="Nom du modèle", 
        description="Nom sous lequel enregistrer le nouveau modèle (optionnel)",
        example="mon_nouveau_modele"
    )
    base_model_name: Optional[str] = Field(
        None, 
        title="Nom du modèle de base", 
        description="Nom du modèle à utiliser comme base (optionnel)",
        example="dst_trustpilot"
    )
    base_model_version: Optional[str] = Field(
        None, 
        title="Version du modèle de base", 
        description="Version du modèle de base à utiliser (optionnel)",
        example="1"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "run_id": "a1b2c3d4e5f6",
                "model_name": "mon_nouveau_modele",
                "base_model_name": "dst_trustpilot",
                "base_model_version": "1"
            }
        }

class TrainingResponse(BaseModel):
    """Réponse d'entraînement du modèle"""
    status: str = Field(
        ..., 
        title="Statut", 
        description="Statut de la requête d'entraînement",
        example="success"
    )
    metrics: Dict[str, float] = Field(
        ..., 
        title="Métriques", 
        description="Métriques d'entraînement et d'évaluation",
        example={"train_accuracy": 0.85, "test_accuracy": 0.82}
    )
    run_id: str = Field(
        ..., 
        title="ID du run MLflow", 
        description="ID du run MLflow d'entraînement",
        example="a1b2c3d4e5f6"
    )
    data_path: str = Field(
        ..., 
        title="Chemin des données", 
        description="Chemin vers les données utilisées pour l'entraînement",
        example="data/processed/processed_data_20250723_120000.csv"
    )
    message: str = Field(
        ..., 
        title="Message", 
        description="Message décrivant le résultat de l'entraînement",
        example="Modèle entraîné avec succès"
    )
    model_name: str = Field(
        ..., 
        title="Nom du modèle", 
        description="Nom du modèle enregistré",
        example="dst_trustpilot"
    )
    model_version: str = Field(
        ..., 
        title="Version du modèle", 
        description="Version du modèle enregistré",
        example="2"
    )

class IngestionResponse(BaseModel):
    """Réponse d'ingestion des données"""
    status: str = Field(
        ..., 
        title="Statut", 
        description="Statut de la requête d'ingestion",
        example="success"
    )
    n_processed_rows: int = Field(
        ..., 
        title="Nombre de lignes traitées", 
        description="Nombre de lignes traitées et conservées",
        example=1000
    )
    stats: Dict = Field(
        ..., 
        title="Statistiques", 
        description="Statistiques sur les données traitées",
        example={
            "n_rows": 1000, 
            "n_missing_avis": 0, 
            "n_missing_notes": 0, 
            "avg_note": 4.2, 
            "min_note": 1, 
            "max_note": 5,
            "avg_avis_length": 120.5
        }
    )
    saved_path: Optional[str] = Field(
        None,
        title="Chemin de sauvegarde",
        description="Chemin où les données traitées ont été sauvegardées",
        example="data/processed/processed_data_20250721_001436.csv"
    )
    
class ValidationRequest(BaseModel):
    """Modèle pour une requête de validation de modèle"""
    model_name: Optional[str] = Field(
        None, 
        title="Nom du modèle", 
        description="Nom du modèle à valider (optionnel, tous les modèles en attente si non spécifié)",
        example="dst_trustpilot"
    )
    model_version: Optional[str] = Field(
        None, 
        title="Version du modèle", 
        description="Version du modèle à valider (obligatoire si model_name est spécifié)",
        example="2"
    )
    auto_approve: bool = Field(
        False, 
        title="Approbation automatique", 
        description="Si True, le modèle sera automatiquement promu en production s'il passe la validation",
        example=False
    )
    threshold: Optional[float] = Field(
        None, 
        title="Seuil de validation", 
        description="Seuil d'accuracy pour considérer le modèle comme validé (utilise la valeur de configuration par défaut)",
        example=0.75
    )
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "dst_trustpilot",
                "model_version": "2",
                "auto_approve": True,
                "threshold": 0.75
            }
        }
    
class ValidationResponse(BaseModel):
    """Réponse de validation de modèle"""
    status: str = Field(
        ..., 
        title="Statut", 
        description="Statut de la requête de validation",
        example="success"
    )
    validation_id: str = Field(
        ..., 
        title="ID de validation", 
        description="Identifiant unique de cette session de validation",
        example="a1b2c3d4"
    )
    models_validated: int = Field(
        ..., 
        title="Nombre de modèles validés", 
        description="Nombre de modèles évalués pendant cette validation",
        example=1
    )
    results: List[Dict[str, Any]] = Field(
        ..., 
        title="Résultats", 
        description="Résultats détaillés de la validation pour chaque modèle",
        example=[{
            "model_name": "dst_trustpilot",
            "model_version": "2",
            "accuracy": 0.82,
            "approved": True,
            "action_taken": "promoted_to_production"
        }]
    )
    saved_path: str = Field(
        ..., 
        title="Chemin de sauvegarde", 
        description="Chemin où les données traitées ont été sauvegardées",
        example="data/processed/processed_data_20250723_120000.csv"
    )

@app.post("/upload", tags=["Données"], summary="Upload de données d'entraînement CSV")
@log_endpoint
async def upload_data(file: UploadFile = File(
    ..., 
    description="Fichier CSV contenant les avis clients à traiter pour l'entraînement. Doit inclure les colonnes 'Avis' et 'Note'."
)):
    """
    Endpoint pour uploader et traiter un fichier CSV d'avis clients.
    
    Le fichier doit contenir au minimum les colonnes 'Avis' et 'Note'.
    
    - 'Avis' : Texte de l'avis client
    - 'Note' : Note numérique (généralement de 1 à 5)
    
    Returns:
        IngestionResponse: Informations sur le traitement effectué
    """
    # Création d'un ID unique pour cet upload
    upload_id = str(uuid.uuid4())[:8]
    logger.info(f"[{upload_id}] Réception d'un fichier: {file.filename}, taille: {file.size} bytes")
    
    try:
        # Vérification de l'extension
        if not file.filename.endswith('.csv'):
            logger.warning(f"[{upload_id}] Extension de fichier non valide: {file.filename}")
            raise HTTPException(status_code=400, detail="Le fichier doit être au format CSV")
        
        # Création d'un dossier temporaire pour stocker le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            logger.info(f"[{upload_id}] Création d'un fichier temporaire: {tmp_file.name}")
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            logger.debug(f"[{upload_id}] Fichier temporaire créé: {tmp_file.name}, taille: {len(content)} bytes")
            
            # Création du pipeline d'ingestion
            logger.info(f"[{upload_id}] Création du pipeline d'ingestion pour données d'entraînement")
            pipeline = DataIngestionPipeline(
                data_path=tmp_file.name,
                experiment_name="data_ingestion_api",
                is_validation_set=False
            )
            
            # Mesure du temps de traitement
            start_time = time.time()
            
            # Exécution du pipeline
            logger.info(f"[{upload_id}] Exécution du pipeline d'ingestion")
            processed_data = pipeline.run_pipeline()
            
            # Calcul du temps d'exécution
            execution_time = time.time() - start_time
            logger.info(f"[{upload_id}] Pipeline exécuté en {execution_time:.3f}s - Lignes traitées: {len(processed_data)}")
            
            # Création du dossier processed s'il n'existe pas
            logger.debug(f"[{upload_id}] Création du dossier de sortie data/processed")
            os.makedirs('data/processed', exist_ok=True)
            
            # Sauvegarde des données traitées
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/processed/processed_data_{timestamp}.csv"
            logger.info(f"[{upload_id}] Sauvegarde des données traitées: {output_path}")
            processed_data.to_csv(output_path, index=False)
            
            # Calcul des statistiques
            logger.debug(f"[{upload_id}] Calcul des statistiques sur les données")
            stats = pipeline.get_data_stats(processed_data)
            logger.info(f"[{upload_id}] Statistiques: {len(processed_data)} lignes, note moyenne: {stats.get('avg_note', 'N/A')}")
            
            response = IngestionResponse(
                status="success",
                n_processed_rows=len(processed_data),
                stats=stats,
                saved_path=output_path
            )
            
            logger.info(f"[{upload_id}] Traitement terminé avec succès: {len(processed_data)} lignes sauvegardées dans {output_path}")
            return response
            
    except Exception as e:
        logger.error(f"[{upload_id}] Erreur lors du traitement du fichier: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Nettoyage du fichier temporaire
        if 'tmp_file' in locals():
            logger.debug(f"[{upload_id}] Nettoyage du fichier temporaire: {tmp_file.name}")
            os.unlink(tmp_file.name)

@app.post("/upload/validation", tags=["Données"], summary="Upload de données de validation CSV")
@log_endpoint
async def upload_validation_data(file: UploadFile = File(
    ..., 
    description="Fichier CSV contenant les avis clients à utiliser comme données de validation. Doit inclure les colonnes 'Avis' et 'Note'."
)):
    """
    Endpoint pour uploader et traiter un fichier CSV d'avis clients spécifiquement pour la validation des modèles.
    
    Le fichier doit contenir au minimum les colonnes 'Avis' et 'Note'.
    Ces données seront taguées comme 'jdd validation' et utilisées pour évaluer les modèles avant leur mise en production.
    
    - 'Avis' : Texte de l'avis client
    - 'Note' : Note numérique (généralement de 1 à 5)
    
    Returns:
        IngestionResponse: Informations sur le traitement effectué
    """
    # Création d'un ID unique pour cet upload
    upload_id = str(uuid.uuid4())[:8]
    logger.info(f"[{upload_id}] Réception d'un fichier de validation: {file.filename}, taille: {file.size} bytes")
    
    try:
        # Vérification de l'extension
        if not file.filename.endswith('.csv'):
            logger.warning(f"[{upload_id}] Extension de fichier non valide: {file.filename}")
            raise HTTPException(status_code=400, detail="Le fichier doit être au format CSV")
        
        # Création d'un dossier temporaire pour stocker le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            logger.info(f"[{upload_id}] Création d'un fichier temporaire: {tmp_file.name}")
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            logger.debug(f"[{upload_id}] Fichier temporaire créé: {tmp_file.name}, taille: {len(content)} bytes")
            
            # Création du pipeline d'ingestion avec is_validation_set=True
            logger.info(f"[{upload_id}] Création du pipeline d'ingestion pour données de validation")
            pipeline = DataIngestionPipeline(
                data_path=tmp_file.name,
                experiment_name="data_ingestion_api",
                is_validation_set=True
            )
            
            # Mesure du temps de traitement
            start_time = time.time()
            processed_data = pipeline.run_pipeline()
            processing_time = time.time() - start_time
            logger.info(f"[{upload_id}] Traitement effectué en {processing_time:.3f}s - {len(processed_data)} lignes")
            
            # Sauvegarde des données traitées
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/processed/validation_data_{timestamp}.csv"
            logger.info(f"[{upload_id}] Sauvegarde des données de validation: {output_path}")
            processed_data.to_csv(output_path, index=False)
            
            # Calcul des statistiques
            logger.debug(f"[{upload_id}] Calcul des statistiques sur les données")
            stats = pipeline.get_data_stats(processed_data)
            logger.info(f"[{upload_id}] Statistiques: {len(processed_data)} lignes, note moyenne: {stats.get('avg_note', 'N/A')}")
            
            response = IngestionResponse(
                status="success",
                n_processed_rows=len(processed_data),
                stats=stats,
                saved_path=output_path
            )
            
            logger.info(f"[{upload_id}] Traitement de données de validation terminé avec succès: {len(processed_data)} lignes sauvegardées dans {output_path}")
            return response
            
    except Exception as e:
        logger.error(f"[{upload_id}] Erreur lors du traitement du fichier de validation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du fichier: {str(e)}")
    finally:
        # Nettoyage du fichier temporaire
        if 'tmp_file' in locals():
            logger.debug(f"[{upload_id}] Nettoyage du fichier temporaire: {tmp_file.name}")
            os.unlink(tmp_file.name)

@app.get("/", tags=["Général"])
@log_endpoint
def read_root():
    """Point d'entrée de l'API"""
    logger.info("Accès à la racine de l'API")
    return {"message": "Bienvenue sur l'API de prédiction"}

@app.post("/predict/form", response_model=PredictionResponse, tags=["Prédiction"], summary="Prédiction via formulaire ✨")
@log_endpoint
def predict_form(
    text: str = Form(
        "Ce produit est vraiment excellent, je le recommande vivement !", 
        description="Le texte de l'avis client pour lequel vous souhaitez prédire le sentiment",
        example="Ce produit est vraiment excellent, je le recommande vivement !"
    ),
    model_name: Optional[str] = Form(
        "", 
        description="Nom du modèle à utiliser pour la prédiction (laisser vide pour utiliser le modèle par défaut)",
        example="dst_trustpilot"
    ),
    model_version: Optional[str] = Form(
        "", 
        description="Version spécifique du modèle à utiliser (laisser vide pour utiliser la dernière version)",
        example="1"
    )
):
    """
    ✅ Endpoint pour la classification de texte via formulaire (recommandé pour les tests)
    
    Un exemple de texte est déjà pré-rempli pour faciliter les tests.
    
    Permet de soumettre facilement :
    - **text**: Le texte de l'avis client à analyser (pré-rempli avec un exemple)
    - **model_name** (optionnel): Laisser vide pour utiliser le modèle par défaut
    - **model_version** (optionnel): Laisser vide pour utiliser la dernière version
        
    Returns:
        PredictionResponse: Prédiction du modèle avec les détails suivants :
            - prediction: 0 pour négatif, 1 pour positif
            - probabilities: probabilités pour chaque classe
            - sentiment: "négatif" ou "positif" en texte
    """
    return _predict_internal(text, model_name, model_version)

@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"], summary="Prédiction via JSON")
def predict(request: PredictionRequest):
    """
    Endpoint pour la classification de texte via JSON (format standard)
    
    Pour une version avec formulaire, utilisez plutôt l'endpoint `/predict/form`
    
    Args:
        request (PredictionRequest): Requête contenant le texte à classifier
        
    Returns:
        PredictionResponse: Prédiction du modèle avec les détails suivants :
            - prediction: 0 pour négatif, 1 pour positif
            - probabilities: probabilités pour chaque classe
            - sentiment: "négatif" ou "positif" en texte
    """
    # Log manuel au lieu d'utiliser le décorateur
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Reçu: POST /predict - Texte: {request.text[:30]}...")
    
    result = _predict_internal(request.text, request.model_name, request.model_version)
    
    logger.info(f"[{request_id}] Terminé: POST /predict - Prédiction: {result.prediction}")
    return result

def _predict_internal(text: str, model_name: Optional[str] = None, model_version: Optional[str] = None):
    """
    Fonction interne de prédiction utilisée par les deux endpoints
    """
    # Création d'un ID unique pour cette prédiction
    pred_id = str(uuid.uuid4())[:8]
    logger.info(f"[{pred_id}] Nouvelle demande de prédiction - Texte: '{text[:50]}...' - Modèle: {model_name or 'défaut'}, Version: {model_version or 'dernière'}")
    
    # Convertir les chaînes vides en None pour le traitement correct des valeurs par défaut
    if model_name == "":
        model_name = None
        logger.debug(f"[{pred_id}] Nom du modèle vide converti en None")
    if model_version == "":
        model_version = None
        logger.debug(f"[{pred_id}] Version du modèle vide convertie en None")
    
    try:
        # Conversion du texte en série pandas
        logger.debug(f"[{pred_id}] Conversion du texte en série pandas")
        text_series = pd.Series([text])
        
        # Prédiction avec le modèle spécifié
        logger.info(f"[{pred_id}] Obtention du service de prédiction")
        service = get_prediction_service(
            model_name=model_name,
            model_version=model_version
        )
        
        # Mesure du temps de prédiction
        start_time = time.time()
        logger.info(f"[{pred_id}] Exécution de la prédiction")
        prediction_proba = service.predict_proba(text_series)
        execution_time = time.time() - start_time
        logger.info(f"[{pred_id}] Prédiction effectuée en {execution_time:.3f}s")
        
        # Le modèle retourne un tableau de forme (1, 2) avec des probabilités
        if not isinstance(prediction_proba, np.ndarray) or prediction_proba.ndim != 2:
            logger.error(f"[{pred_id}] Format de prédiction invalide: {type(prediction_proba)}, shape: {getattr(prediction_proba, 'shape', 'N/A')}")
            raise ValueError("Format de prédiction invalide")
            
        # Extraire les probabilités
        neg_proba, pos_proba = prediction_proba[0]
        logger.debug(f"[{pred_id}] Probabilités: négatif={neg_proba:.4f}, positif={pos_proba:.4f}")
        
        # Déterminer la classe prédite
        predicted_class = 1 if pos_proba > neg_proba else 0
        sentiment = "positif" if predicted_class == 1 else "négatif"
        logger.info(f"[{pred_id}] Classe prédite: {predicted_class} ({sentiment})")
        
        response = PredictionResponse(
            prediction=predicted_class,
            probabilities={
                "négatif": float(neg_proba),
                "positif": float(pos_proba)
            },
            sentiment=sentiment
        )
        
        logger.info(f"[{pred_id}] Prédiction terminée avec succès: {sentiment} (score: {max(neg_proba, pos_proba):.4f})")
        return response
    
    except Exception as e:
        logger.error(f"[{pred_id}] Erreur lors de la prédiction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train/form", response_model=TrainingResponse, tags=["Entraînement"], summary="Entraînement via formulaire ✨")
@log_endpoint
async def train_model_form(
    run_id: Optional[str] = Form(
        "",
        description="ID du run MLflow contenant les données d'entraînement (laisser vide pour utiliser le dernier run)",
        example="a1b2c3d4e5f6"
    ),
    model_name: Optional[str] = Form(
        "",
        description="Nom sous lequel enregistrer le nouveau modèle (laisser vide pour utiliser le nom par défaut)",
        example="mon_nouveau_modele"
    ),
    base_model_name: Optional[str] = Form(
        "",
        description="Nom du modèle à utiliser comme base (laisser vide pour utiliser le modèle par défaut)",
        example="dst_trustpilot"
    ),
    base_model_version: Optional[str] = Form(
        "",
        description="Version du modèle de base à utiliser (laisser vide pour utiliser la dernière version)",
        example="1"
    )
):
    """
    ✅ Endpoint pour entraîner le modèle sur de nouvelles données via formulaire (recommandé pour les tests).
    Utilise les données d'un run MLflow d'ingestion spécifique ou le dernier run réussi.
    
    Tous les champs sont optionnels et peuvent être laissés vides.
    
    Le formulaire permet de soumettre :
    - **run_id** (optionnel): Laisser vide pour utiliser le dernier run d'ingestion
    - **model_name** (optionnel): Laisser vide pour utiliser un nom généré automatiquement
    - **base_model_name** (optionnel): Laisser vide pour utiliser le modèle par défaut
    - **base_model_version** (optionnel): Laisser vide pour utiliser la dernière version
        
    Returns:
        TrainingResponse: Résultat de l'entraînement avec les métriques et informations
    """
    return await _train_internal(run_id, model_name, base_model_name, base_model_version)

@app.post("/train", response_model=TrainingResponse, tags=["Entraînement"], summary="Entraînement via JSON")
@log_endpoint
async def train_model(request: TrainingRequest):
    """
    Endpoint pour entraîner le modèle sur de nouvelles données via JSON.
    Utilise les données d'un run MLflow d'ingestion spécifique ou le dernier run réussi.
    
    Pour une version avec formulaire, utilisez plutôt l'endpoint `/train/form`
    
    Args:
        request (TrainingRequest): Requête contenant optionnellement l'ID du run MLflow
        
    Returns:
        TrainingResponse: Résultat de l'entraînement avec les métriques et informations
    """
    return await _train_internal(
        request.run_id, 
        request.model_name, 
        request.base_model_name, 
        request.base_model_version
    )

async def _train_internal(
    run_id: Optional[str] = None,
    model_name: Optional[str] = None,
    base_model_name: Optional[str] = None,
    base_model_version: Optional[str] = None
):
    """
    Fonction interne pour l'entraînement du modèle utilisée par les deux endpoints
    """
    # Création d'un ID unique pour cet entraînement
    train_id = str(uuid.uuid4())[:8]
    logger.info(f"[{train_id}] Nouvelle demande d'entraînement - Run ID: {run_id or 'Auto'}, Modèle: {model_name or 'Auto'}")
    
    # Convertir les chaînes vides en None pour le traitement correct des valeurs par défaut
    if run_id == "":
        run_id = None
        logger.debug(f"[{train_id}] Run ID vide converti en None")
    if model_name == "":
        model_name = None
        logger.debug(f"[{train_id}] Nom du modèle vide converti en None")
    if base_model_name == "":
        base_model_name = None
        logger.debug(f"[{train_id}] Nom du modèle de base vide converti en None")
    if base_model_version == "":
        base_model_version = None
        logger.debug(f"[{train_id}] Version du modèle de base vide convertie en None")
        
    try:
        # Lancement de l'entraînement
        logger.info(f"[{train_id}] Démarrage de l'entraînement{'avec run_id: ' + run_id if run_id else ''}")
        
        # Mesure du temps d'entraînement
        start_time = time.time()
        
        metrics = train_model_function(
            run_id=run_id,
            model_name=model_name,
            base_model_name=base_model_name,
            base_model_version=base_model_version
        )
        
        # Calcul du temps d'entraînement
        execution_time = time.time() - start_time
        logger.info(f"[{train_id}] Entraînement terminé en {execution_time:.3f}s")
        
        # Log des métriques obtenues
        logger.info(f"[{train_id}] Métriques: Train accuracy={metrics['train_accuracy']:.4f}, Test accuracy={metrics['test_accuracy']:.4f}")
        
        # Récupération du run MLflow actuel
        run = mlflow.get_run(run_id=metrics["run_id"])
        if not run:
            logger.error(f"[{train_id}] Impossible de récupérer le run MLflow pour l'ID: {metrics['run_id']}")
            raise ValueError("Impossible de récupérer le run MLflow")
            
        logger.info(f"[{train_id}] Modèle entraîné et enregistré - Run ID: {run.info.run_id}, Modèle: {metrics['model_name']}, Version: {metrics['model_version']}")
        
        return TrainingResponse(
            status="success",
            metrics={
                "train_accuracy": metrics["train_accuracy"],
                "test_accuracy": metrics["test_accuracy"]
            },
            run_id=run.info.run_id,
            data_path=metrics["data_path"],
            message="Modèle entraîné avec succès",
            model_name=metrics["model_name"],
            model_version=metrics["model_version"]
        )
        
    except Exception as e:
        logger.error(f"[{train_id}] Erreur lors de l'entraînement: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'entraînement: {str(e)}"
        )

@app.post("/validate", response_model=ValidationResponse, tags=["Validation"], summary="Validation de modèle via JSON")
@log_endpoint
async def validate(request: ValidationRequest):
    """
    Endpoint pour valider un ou plusieurs modèles.
    
    Si model_name et model_version sont spécifiés, valide uniquement ce modèle.
    Sinon, valide tous les modèles en attente de validation (marqués "à valider").
    
    Si auto_approve=True, les modèles qui passent la validation sont automatiquement promus en production.
    
    Args:
        request (ValidationRequest): Paramètres de validation
        
    Returns:
        ValidationResponse: Résultats de la validation
    """
    validation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{validation_id}] Nouvelle demande de validation - Modèle: {request.model_name or 'tous'}, "
               f"Version: {request.model_version or 'toutes'}, Auto-approbation: {request.auto_approve}")
    
    # Exécution de la validation
    result = validate_model(
        model_name=request.model_name,
        model_version=request.model_version,
        approve=request.auto_approve,
        threshold=request.threshold
    )
    
    logger.info(f"[{validation_id}] Validation terminée - {result['models_validated']} modèles validés")
    return result

@app.post("/validate/form", response_model=ValidationResponse, tags=["Validation"], summary="Validation de modèle via formulaire ✨")
@log_endpoint
async def validate_form(
    model_name: Optional[str] = Form(
        "", 
        description="Nom du modèle à valider (laisser vide pour tous les modèles en attente)",
        example="dst_trustpilot"
    ),
    model_version: Optional[str] = Form(
        "", 
        description="Version du modèle à valider (obligatoire si un nom de modèle est spécifié)",
        example="2"
    ),
    auto_approve: bool = Form(
        False, 
        description="Si coché, les modèles validés seront automatiquement promus en production",
        example=False
    ),
    threshold: Optional[float] = Form(
        None, 
        description="Seuil d'accuracy pour la validation (utilise la valeur par défaut si non spécifié)",
        example=0.75
    )
):
    """
    Endpoint pour valider un ou plusieurs modèles via formulaire.
    
    Si model_name et model_version sont spécifiés, valide uniquement ce modèle.
    Sinon, valide tous les modèles en attente de validation (marqués "à valider").
    
    Si auto_approve=True, les modèles qui passent la validation sont automatiquement promus en production.
    
    Returns:
        ValidationResponse: Résultats de la validation
    """
    # Convertir les chaînes vides en None
    if model_name == "":
        model_name = None
    if model_version == "":
        model_version = None
    
    validation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{validation_id}] Nouvelle demande de validation (formulaire) - Modèle: {model_name or 'tous'}, "
               f"Version: {model_version or 'toutes'}, Auto-approbation: {auto_approve}")
    
    # Exécution de la validation
    result = validate_model(
        model_name=model_name,
        model_version=model_version,
        approve=auto_approve,
        threshold=threshold
    )
    
    logger.info(f"[{validation_id}] Validation terminée - {result['models_validated']} modèles validés")
    return result

@app.post("/promote/{model_name}/{model_version}", response_model=ValidationResponse, tags=["Validation"], summary="Promotion de modèle en production")
@log_endpoint
async def promote_model(
    model_name: str = Path(..., description="Nom du modèle à promouvoir en production"),
    model_version: str = Path(..., description="Version du modèle à promouvoir en production")
):
    """
    Endpoint pour valider et promouvoir directement un modèle en production.
    
    Le modèle sera d'abord validé et, s'il passe la validation avec succès, sera promu en production.
    
    Args:
        model_name: Nom du modèle à promouvoir
        model_version: Version du modèle à promouvoir
        
    Returns:
        ValidationResponse: Résultat de la validation et promotion
    """
    promotion_id = str(uuid.uuid4())[:8]
    logger.info(f"[{promotion_id}] Demande de promotion directe - Modèle: {model_name}, Version: {model_version}")
    
    # Validation et promotion
    result = validate_and_promote_model(model_name, model_version)
    
    if result['results'] and result['results'][0].get('action_taken') == 'promoted_to_production':
        logger.info(f"[{promotion_id}] Modèle {model_name} v{model_version} promu en production avec succès")
    else:
        logger.warning(f"[{promotion_id}] Échec de la promotion du modèle {model_name} v{model_version}")
    
    return result

if __name__ == "__main__":
    import uvicorn
    
    # Le logging est déjà initialisé au début du fichier
    logger.info("Démarrage du serveur API")
    
    # Démarrage du serveur
    uvicorn.run(app, host="0.0.0.0", port=8042)  # Port 8042 pour correspondre au Dockerfile
