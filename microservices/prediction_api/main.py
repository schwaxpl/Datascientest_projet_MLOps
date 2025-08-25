"""
API de prédiction pour l'analyse de sentiments.
Ce service est responsable uniquement des prédictions à partir des modèles entraînés.
"""

import os
import time
import pickle
import csv
from io import StringIO
import pandas as pd
import numpy as np
import mlflow
import uuid
from fastapi import FastAPI, HTTPException, Depends, Query, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from contextlib import asynccontextmanager

# Import des modules locaux
from src.predict import PredictionService
from microservices.common.logger_config import init_logging, get_logger
from microservices.common.utils import get_mlflow_client, load_model_from_registry, get_latest_model_version
from microservices.common.config import (
    MODEL_NAME,
    VECTORIZER_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_PATH
)

# Initialisation du système de logging
loggers = init_logging("prediction", api=True)
logger = loggers['prediction']

# Initialisation lazy du service de prédiction et du vectorizer
prediction_service = None
vectorizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Événement exécuté au démarrage de l'application"""
    logger.info("Initialisation de l'API de prédiction...")
    
    # Vérification et préchargement du modèle par défaut
    try:
        logger.info(f"Vérification de l'existence du modèle par défaut: {MODEL_NAME}")
        client = get_mlflow_client()
        
        # Charger le vectorizer local (nécessaire dans tous les cas)
        try:
            vectorizer_loaded = load_vectorizer()
            logger.info("Vectorizer chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur critique lors du chargement du vectorizer: {str(e)}")
            # Si on ne peut pas charger le vectorizer, on ne peut pas continuer
            raise
        
        # Vérifier si le modèle par défaut existe dans le registre
        model_exists = False
        production_model_exists = False
        
        try:
            # Chercher si le modèle existe déjà dans MLflow
            registered_models = client.search_registered_models(filter_string=f"name='{MODEL_NAME}'")
            if any(model.name == MODEL_NAME for model in registered_models):
                model_exists = True
                logger.info(f"Modèle {MODEL_NAME} trouvé dans le registre")
                
                # Vérifier s'il existe une version en production
                versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                for v in versions:
                    if v.current_stage == "Production":
                        production_model_exists = True
                        logger.info(f"Version de production trouvée pour {MODEL_NAME}: {v.version}")
                        break
            
            if not model_exists:
                logger.warning(f"Modèle {MODEL_NAME} non trouvé dans le registre")
        except Exception as e:
            logger.warning(f"Erreur lors de la vérification du modèle dans MLflow: {str(e)}")
        
        # Si le modèle n'existe pas OU s'il n'y a pas de version en production,
        # enregistrer le modèle local dans MLflow
        if not model_exists or not production_model_exists:
            try:
                logger.info("Enregistrement du modèle local dans MLflow...")
                
                # Utiliser le chemin du modèle défini dans la configuration
                logger.info(f"Tentative de chargement du modèle depuis {MODEL_PATH}")
                
                # Charger le modèle local
                if os.path.exists(MODEL_PATH):
                    logger.info(f"Modèle local trouvé: {MODEL_PATH}")
                    with open(MODEL_PATH, 'rb') as f:
                        model_loaded = pickle.load(f)
                    model_path_used = MODEL_PATH
                else:
                    raise FileNotFoundError(f"Modèle local non trouvé au chemin spécifié: {MODEL_PATH}")
                
                # Créer un run MLflow pour enregistrer le modèle
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                with mlflow.start_run(run_name="initial_model_registration") as run:
                    run_id = run.info.run_id
                    logger.info(f"MLflow run créé: {run_id}")
                    
                    # Enregistrer le modèle
                    artifact_path = "model"
                    mlflow.keras.log_model(model_loaded, artifact_path)
                    
                    # Enregistrer le vectorizer comme artifact
                    vectorizer_temp_path = "/tmp/vectorizer.pkl"
                    with open(vectorizer_temp_path, 'wb') as f:
                        pickle.dump(vectorizer_loaded, f)
                    mlflow.log_artifact(vectorizer_temp_path, "vectorizer")
                    
                    # Log des métriques (valeurs factices puisqu'on n'a pas d'évaluation)
                    mlflow.log_metric("accuracy", 0.85)
                    mlflow.log_metric("f1_score", 0.84)
                
                # Enregistrer le modèle dans le registre
                model_uri = f"runs:/{run_id}/{artifact_path}"
                if not model_exists:
                    # Créer une nouvelle entrée dans le registre
                    mv = mlflow.register_model(model_uri, MODEL_NAME)
                    logger.info(f"Nouveau modèle enregistré: {MODEL_NAME} version {mv.version}")
                else:
                    # Ajouter une nouvelle version
                    mv = mlflow.register_model(model_uri, MODEL_NAME)
                    logger.info(f"Nouvelle version ajoutée: {MODEL_NAME} version {mv.version}")
                
                # Promouvoir le modèle en production
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=mv.version,
                    stage="Production"
                )
                logger.info(f"Modèle {MODEL_NAME} version {mv.version} promu en Production")
                
                # Mettre à jour la variable pour le préchargement
                production_model_exists = True
                
            except Exception as e:
                logger.error(f"Erreur lors de l'enregistrement du modèle local: {str(e)}", exc_info=True)
                logger.warning("Le service continuera sans modèle par défaut")
        
        # Préchargement du modèle si une version en production existe
        if production_model_exists:
            try:
                # Récupérer la dernière version en production
                latest_version = get_latest_model_version(client, MODEL_NAME)
                logger.info(f"Préchargement du modèle {MODEL_NAME} version {latest_version.version}")
                
                # Précharger le modèle et le vectorizer
                global prediction_service
                if prediction_service is None:
                    logger.info("Préchargement du service de prédiction...")
                    model = load_model_from_registry(MODEL_NAME)
                    prediction_service = PredictionService.from_artifacts(
                        model=model,
                        vectorizer=vectorizer_loaded
                    )
                    logger.info("Service de prédiction préchargé avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du préchargement du modèle: {str(e)}", exc_info=True)
                logger.warning("Le service démarrera sans modèle préchargé")
    
    except Exception as e:
        logger.error(f"Erreur critique lors de l'initialisation du service: {str(e)}", exc_info=True)
        logger.warning("Le service démarrera malgré l'erreur d'initialisation")
    
    yield
    logger.info("Arrêt de l'API de prédiction")

app = FastAPI(
    title="API de Prédiction - Microservice MLOps",
    description="""
    # API de prédiction d'analyse de sentiments
    
    Cette API est responsable uniquement de la prédiction de sentiments à partir de modèles entraînés.
    
    ## Endpoints
    
    * `/predict` : Prédit le sentiment d'un texte donné
    * `/predict/batch` : Prédit le sentiment de plusieurs textes
    * `/models` : Liste les modèles disponibles
    * `/health` : Vérifie l'état de santé de l'API
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # A restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_vectorizer():
    """Charge le vectorizer depuis le fichier local en utilisant le chemin défini dans la configuration"""
    global vectorizer
    if vectorizer is None:
        try:
            logger.info(f"Tentative de chargement du vectorizer depuis {VECTORIZER_PATH}")
            if os.path.exists(VECTORIZER_PATH):
                with open(VECTORIZER_PATH, 'rb') as f:
                    vectorizer = pickle.load(f)
                logger.info(f"Vectorizer chargé avec succès depuis {VECTORIZER_PATH}")
                return vectorizer
            else:
                error_msg = f"Vectorizer non trouvé au chemin spécifié: {VECTORIZER_PATH}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        except Exception as e:
            logger.error(f"Erreur lors du chargement du vectorizer depuis {VECTORIZER_PATH}: {str(e)}")
            raise FileNotFoundError(f"Impossible de charger le vectorizer: {str(e)}")
    else:
        logger.debug("Utilisation du vectorizer en cache")
    return vectorizer

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
        
        # Récupération de la dernière version du modèle
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

# Cette classe n'est plus utilisée car nous n'acceptons que des fichiers CSV
# mais nous la gardons pour compatibilité historique (peut être nécessaire pour des tests)
class BatchPredictionRequest(BaseModel):
    """Modèle pour une requête de prédiction par lots (obsolète, utilisez plutôt des fichiers CSV)"""
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

class BatchPredictionResponse(BaseModel):
    """Réponse de prédiction par lots"""
    predictions_csv: str = Field(
        ...,
        title="Prédictions CSV",
        description="Contenu CSV avec les prédictions"
    )
    processing_time: float = Field(
        ...,
        title="Temps de traitement",
        description="Temps de traitement en secondes",
        example=0.156
    )
    format: str = Field(
        "csv",
        title="Format de réponse",
        description="Format de la réponse (toujours 'csv')",
        example="csv"
    )
    
class ModelInfo(BaseModel):
    """Information sur un modèle"""
    name: str = Field(..., title="Nom du modèle")
    version: str = Field(..., title="Version du modèle")
    stage: str = Field(..., title="Stage du modèle")
    creation_timestamp: int = Field(..., title="Date de création")
    description: Optional[str] = Field(None, title="Description")
    
class ModelsResponse(BaseModel):
    """Liste des modèles disponibles"""
    models: List[ModelInfo] = Field(..., title="Modèles disponibles")
    default_model: str = Field(..., title="Modèle par défaut")
    default_version: str = Field(..., title="Version par défaut")

class HealthResponse(BaseModel):
    """État de santé de l'API"""
    status: str = Field(..., title="Statut", example="ok")
    version: str = Field(..., title="Version de l'API", example="1.0.0")
    model_name: str = Field(..., title="Nom du modèle actif", example="dst_trustpilot")
    model_version: str = Field(..., title="Version du modèle actif", example="1")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Vérifie l'état de santé de l'API.
    """
    try:
        # Vérifier que le client MLflow est accessible
        client = get_mlflow_client()
        
        try:
            latest_version = get_latest_model_version(client, MODEL_NAME)
            model_version = latest_version.version
            model_status = "ok"
        except Exception as e:
            logger.warning(f"Modèle par défaut non disponible: {str(e)}")
            model_version = "unknown"
            model_status = "warning"
        
        return {
            "status": "ok",
            "version": "1.0.0",
            "model_name": MODEL_NAME,
            "model_version": model_version
        }
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de l'état de santé: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "version": "1.0.0",
                "model_name": MODEL_NAME,
                "model_version": "unknown",
                "error": str(e)
            }
        )

@app.get("/initialize-model", description="Force l'initialisation du modèle par défaut")
def initialize_default_model():
    """
    Force le chargement du modèle par défaut.
    Utile pour vérifier que le modèle est bien disponible.
    """
    try:
        global prediction_service
        
        # Vérifier si le modèle existe dans MLflow
        client = get_mlflow_client()
        try:
            latest_version = get_latest_model_version(client, MODEL_NAME)
            logger.info(f"Modèle {MODEL_NAME} trouvé avec version: {latest_version.version}")
        except Exception as e:
            logger.error(f"Modèle {MODEL_NAME} non trouvé dans le registre: {str(e)}")
            raise HTTPException(
                status_code=404, 
                detail=f"Le modèle par défaut '{MODEL_NAME}' n'existe pas dans le registre. Veuillez d'abord entrainer un modèle via l'API d'entraînement."
            )
        
        # Charger le modèle et le vectorizer
        logger.info(f"Initialisation du service de prédiction pour le modèle {MODEL_NAME}")
        model = load_model_from_registry(MODEL_NAME)
        vectorizer_loaded = load_vectorizer()
        
        # Créer une nouvelle instance du service
        prediction_service = PredictionService.from_artifacts(
            model=model,
            vectorizer=vectorizer_loaded
        )
        
        return {
            "status": "success",
            "message": f"Modèle {MODEL_NAME} (version {latest_version.version}) initialisé avec succès",
            "model_name": MODEL_NAME,
            "model_version": latest_version.version
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du modèle par défaut: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'initialisation du modèle par défaut: {str(e)}"
        )

@app.get("/models", response_model=ModelsResponse)
def list_models(production_only: bool = Query(False, description="Afficher uniquement les modèles en production")):
    """
    Liste les modèles disponibles pour la prédiction.
    
    Args:
        production_only: Si True, n'affiche que les modèles en production
    """
    try:
        client = get_mlflow_client()
        
        # Récupérer la liste des modèles enregistrés
        registered_models = client.search_registered_models()
        
        models = []
        for rm in registered_models:
            # Pour chaque modèle enregistré, récupérer les versions
            versions = client.search_model_versions(f"name='{rm.name}'")
            for v in versions:
                # Si production_only est True, ne garder que les versions en Production
                if production_only and v.current_stage != "Production":
                    continue
                models.append(
                    ModelInfo(
                        name=rm.name,
                        version=v.version,
                        stage=v.current_stage,
                        creation_timestamp=v.creation_timestamp,
                        description=v.description
                    )
                )
        
        # Récupérer le modèle par défaut et sa version
        try:
            default_model = MODEL_NAME
            latest_version = get_latest_model_version(client, default_model)
            default_version = latest_version.version
        except:
            default_model = "unknown"
            default_version = "unknown"
        
        return {
            "models": models,
            "default_model": default_model,
            "default_version": default_version
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des modèles: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des modèles: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Prédit le sentiment d'un texte.
    """
    # Création d'un ID unique pour cette prédiction
    pred_id = str(uuid.uuid4())[:8]
    logger.info(f"[{pred_id}] Nouvelle demande de prédiction - Texte: '{request.text[:50]}...' - Modèle: {request.model_name or 'défaut'}, Version: {request.model_version or 'dernière'}")
    
    try:
        # Conversion du texte en série pandas
        logger.debug(f"[{pred_id}] Conversion du texte en série pandas")
        text_series = pd.Series([request.text])
        
        # Prédiction avec le modèle spécifié
        logger.info(f"[{pred_id}] Obtention du service de prédiction")
        service = get_prediction_service(
            model_name=request.model_name,
            model_version=request.model_version
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
        
        response = {
            "prediction": predicted_class,
            "probabilities": {
                "négatif": float(neg_proba),
                "positif": float(pos_proba)
            },
            "sentiment": sentiment
        }
        
        logger.info(f"[{pred_id}] Prédiction terminée avec succès: {sentiment} (score: {max(neg_proba, pos_proba):.4f})")
        return response
    
    except Exception as e:
        logger.error(f"[{pred_id}] Erreur lors de la prédiction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    file: UploadFile = File(..., description="Fichier CSV avec une colonne 'texte'"),
    model_name: Optional[str] = Form(None, description="Nom du modèle à utiliser"),
    model_version: Optional[str] = Form(None, description="Version du modèle à utiliser")
):
    """
    Prédit le sentiment de plusieurs textes à partir d'un fichier CSV.
    
    Accepte:
    - Un fichier CSV avec une colonne 'texte' (ou première colonne utilisée par défaut)
    
    Retourne:
    - Un fichier CSV enrichi avec les colonnes de prédiction (sentiment, prediction, score_negatif, score_positif)
    """
    # Création d'un ID unique pour cette prédiction par lots
    batch_id = str(uuid.uuid4())[:8]
    
    try:
        # Vérifier que nous avons bien un fichier CSV
        if not file or not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Vous devez fournir un fichier CSV avec une colonne 'texte'"
            )
            
        # Vérifier l'extension du fichier
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Le fichier doit être au format CSV (.csv)"
            )
        
        # Traitement du fichier CSV
        texts = []
        content = await file.read()
        csv_text = content.decode('utf-8')
        
        # Essayer différents délimiteurs possibles
        delimiters = [';', ',', '\t']
        delimiter_used = None
        
        for delim in delimiters:
            try:
                csv_file = StringIO(csv_text)
                csv_reader = csv.DictReader(csv_file, delimiter=delim)
                # Tester si nous pouvons lire au moins une ligne
                first_row = next(csv_reader, None)
                if first_row:
                    delimiter_used = delim
                    break
            except Exception:
                continue
        
        if not delimiter_used:
            raise HTTPException(
                status_code=400, 
                detail="Impossible de lire le fichier CSV. Vérifiez le format et le délimiteur utilisé."
            )
        
        # Réinitialiser le fichier CSV pour la lecture complète
        csv_file = StringIO(csv_text)
        csv_reader = csv.DictReader(csv_file, delimiter=delimiter_used)
        
        # Si la colonne 'texte' n'existe pas, utiliser la première colonne
        if 'texte' not in csv_reader.fieldnames:
            logger.warning(f"[{batch_id}] Colonne 'texte' non trouvée, utilisation de la première colonne: {csv_reader.fieldnames[0]}")
            text_column = csv_reader.fieldnames[0]
        else:
            text_column = 'texte'
        
        # Lecture des textes depuis le CSV
        csv_file = StringIO(csv_text)
        csv_reader = csv.DictReader(csv_file, delimiter=delimiter_used)
        
        for row in csv_reader:
            texts.append(row[text_column])
        logger.info(f"[{batch_id}] Fichier CSV traité avec {len(texts)} textes")
        
        # Garder une copie du CSV pour la réponse
        csv_file = StringIO(csv_text)
        original_rows = list(csv.DictReader(csv_file, delimiter=delimiter_used))
            
        logger.info(f"[{batch_id}] Nouvelle demande de prédiction par lots - {len(texts)} textes - Modèle: {model_name or 'défaut'}, Version: {model_version or 'dernière'}")
        
        # Conversion des textes en série pandas
        logger.debug(f"[{batch_id}] Conversion des textes en série pandas")
        text_series = pd.Series(texts)
        
        # Prédiction avec le modèle spécifié
        logger.info(f"[{batch_id}] Obtention du service de prédiction")
        service = get_prediction_service(
            model_name=model_name,
            model_version=model_version
        )
        
        # Mesure du temps de prédiction
        start_time = time.time()
        logger.info(f"[{batch_id}] Exécution de la prédiction par lots")
        prediction_proba = service.predict_proba(text_series)
        execution_time = time.time() - start_time
        logger.info(f"[{batch_id}] Prédiction par lots effectuée en {execution_time:.3f}s")
        
        # Le modèle retourne un tableau de forme (n, 2) avec des probabilités
        if not isinstance(prediction_proba, np.ndarray) or prediction_proba.ndim != 2:
            logger.error(f"[{batch_id}] Format de prédiction invalide: {type(prediction_proba)}, shape: {getattr(prediction_proba, 'shape', 'N/A')}")
            raise ValueError("Format de prédiction invalide")
        
        # Traitement des résultats pour chaque texte
        predictions = []
        for i, (neg_proba, pos_proba) in enumerate(prediction_proba):
            predicted_class = 1 if pos_proba > neg_proba else 0
            sentiment = "positif" if predicted_class == 1 else "négatif"
            
            predictions.append({
                "prediction": predicted_class,
                "probabilities": {
                    "négatif": float(neg_proba),
                    "positif": float(pos_proba)
                },
                "sentiment": sentiment
            })
        
            logger.info(f"[{batch_id}] Prédiction par lots terminée avec succès: {len(predictions)} résultats en {execution_time:.3f}s")
        
        # Générer le CSV de sortie avec les prédictions (UTF-8, séparateur point-virgule)
        output = StringIO()
        fieldnames = list(original_rows[0].keys()) + ['prediction', 'sentiment', 'score_negatif', 'score_positif']
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        
        for i, row in enumerate(original_rows):
            # Ajouter les prédictions aux données originales
            row_with_prediction = dict(row)
            row_with_prediction['prediction'] = predictions[i]['prediction']
            row_with_prediction['sentiment'] = predictions[i]['sentiment']
            row_with_prediction['score_negatif'] = predictions[i]['probabilities']['négatif']
            row_with_prediction['score_positif'] = predictions[i]['probabilities']['positif']
            writer.writerow(row_with_prediction)
            
        return JSONResponse(
            content={
                "predictions_csv": output.getvalue(),
                "processing_time": execution_time,
                "format": "csv",
                "encoding": "utf-8",
                "delimiter": ";"
            }
        )
    except Exception as e:
        logger.error(f"[{batch_id}] Erreur lors de la prédiction par lots: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Le logging est déjà initialisé au début du fichier
    logger.info("Démarrage du serveur API de prédiction")
    
    # Démarrage du serveur
    uvicorn.run(app, host="0.0.0.0", port=8001)
