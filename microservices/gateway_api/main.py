"""
API Passerelle (Gateway) pour l'architecture microservices MLOps.
Responsable de l'authentification et du routage des requêtes vers les services appropriés.
"""

import os
import time
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Request, status, Form, Path, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from jose import JWTError, jwt
from contextlib import asynccontextmanager

# Import des modules locaux
from microservices.common.logger_config import init_logging, get_logger
from microservices.common.utils import (
    call_service, 
    call_service_stream,
    get_user, 
    authenticate_user, 
    create_access_token, 
    get_current_active_user
)
from microservices.common.config import (
    SECRET_KEY, 
    ALGORITHM, 
    ACCESS_TOKEN_EXPIRE_MINUTES,
    PREDICTION_API_URL,
    TRAINING_API_URL,
    DATA_API_URL
)

# Initialisation du système de logging
loggers = init_logging("gateway", api=True)
logger = loggers['gateway']

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Événement exécuté au démarrage de l'application"""
    logger.info("Initialisation de l'API Passerelle...")
    yield
    logger.info("Arrêt de l'API Passerelle")

app = FastAPI(
    title="API Passerelle MLOps",
    description="""
    # API Passerelle pour l'architecture microservices MLOps
    
    Cette API est responsable de :
    * **Authentification** des utilisateurs
    * **Routage** des requêtes vers les services appropriés
    * **Centralisation** du point d'accès aux fonctionnalités
    
    ## Authentification
    
    Utilisez l'endpoint `/token` pour vous authentifier et obtenir un jeton JWT.
    Ce jeton doit être inclus dans l'en-tête `Authorization` de toutes les requêtes.
    
    ## Services disponibles
    
    * **Prédiction** : Analyse de sentiments d'avis clients
    * **Données** : Gestion des jeux de données, équilibrage des classes déséquilibrées
    * **Entraînement** : Entraînement et validation de modèles
    * **Administration** : Gestion des utilisateurs et des services
    
    ## Développé dans le cadre du projet MLOps Datascientest.
    """,
    version="1.0.0",
    lifespan=lifespan,
    # Organisation des tags pour Swagger UI
    openapi_tags=[
        {
            "name": "Authentication",
            "description": "Endpoints liés à l'authentification et la gestion des utilisateurs"
        },
        {
            "name": "System",
            "description": "Endpoints système comme health check et documentation"
        },
        {
            "name": "Prediction",
            "description": "Endpoints liés aux prédictions et aux modèles de prédiction"
        },
        {
            "name": "Data",
            "description": "Endpoints liés à la gestion des données et des datasets"
        },
        {
            "name": "Training",
            "description": "Endpoints liés à l'entraînement et la gestion des modèles"
        }
    ],
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # A restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles de données

class Token(BaseModel):
    """Modèle pour un token JWT"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    user_info: Dict[str, Any]

class UserInfo(BaseModel):
    """Modèle pour les informations utilisateur"""
    username: str
    roles: List[str]

class PredictRequest(BaseModel):
    """Modèle pour une requête de prédiction"""
    text: str = Field(..., title="Texte à analyser", description="Le texte dont on veut prédire le sentiment", example="Ce jeu est vraiment fantastique, j'adore y jouer!")
    model_name: Optional[str] = Field(None, title="Nom du modèle", description="Nom du modèle à utiliser (optionnel, utilisera le modèle par défaut si non spécifié)", example="sentiment_model")

class PredictBatchRequest(BaseModel):
    """Modèle pour une requête de prédiction par lots"""
    texts: List[str] = Field(..., title="Liste de textes", description="Liste des textes à analyser", example=["Ce jeu est vraiment fantastique!", "Ce jeu est horrible, je n'aime pas du tout."])
    model_name: Optional[str] = Field(None, title="Nom du modèle", description="Nom du modèle à utiliser (optionnel, utilisera le modèle par défaut si non spécifié)", example="sentiment_model")

class PredictResponse(BaseModel):
    """Modèle pour une réponse de prédiction"""
    text: str
    sentiment: str
    score: float
    model_info: Dict[str, Any]

# Note: Cette classe n'est plus utilisée car l'endpoint retourne directement un fichier CSV
class PredictBatchResponse(BaseModel):
    """Modèle pour une réponse de prédiction par lots (obsolète, l'endpoint retourne un CSV)"""
    predictions_csv: str = Field(..., title="Contenu CSV avec les prédictions")

class TrainRequest(BaseModel):
    """Modèle pour une requête d'entraînement"""
    run_id: Optional[str] = Field(None, title="ID du run MLflow", description="ID du run MLflow contenant les données d'entraînement", example="a1b2c3d4e5f6")
    model_name: Optional[str] = Field(None, title="Nom du modèle", description="Nom à donner au modèle entraîné", example="mon_nouveau_modele")
    base_model_name: Optional[str] = Field(None, title="Nom du modèle de base", description="Nom du modèle à utiliser comme base", example="dst_trustpilot")
    base_model_version: Optional[str] = Field(None, title="Version du modèle de base", description="Version du modèle de base à utiliser", example="1")

class TraceabilityInfo(BaseModel):
    """Informations de traçabilité"""
    run_id: str = Field(default="unknown", title="ID du Run", description="ID effectivement utilisé")
    source: str = Field(default="unknown", title="Source", description="Origine de la sélection (spécifié ou auto-détecté)")

class BaseModelInfo(BaseModel):
    """Informations sur le modèle de base"""
    name: str = Field(default="unknown", title="Nom", description="Nom du modèle de base")
    version: str = Field(default="unknown", title="Version", description="Version du modèle de base")
    source: str = Field(default="unknown", title="Source", description="Origine de la sélection (spécifié ou auto-détecté)")

class TraceabilityData(BaseModel):
    """Données de traçabilité complètes"""
    ingestion: TraceabilityInfo = Field(..., title="Ingestion", description="Détails sur la source des données")
    base_model: BaseModelInfo = Field(..., title="Modèle de base", description="Détails sur le modèle de base utilisé")

class TrainResponse(BaseModel):
    """Modèle pour une réponse d'entraînement"""
    status: str = Field(..., title="Statut", description="Statut de la requête d'entraînement", example="success")
    metrics: Dict[str, Any] = Field(..., title="Métriques", description="Métriques d'entraînement et d'évaluation", example={"train_accuracy": 0.85, "test_accuracy": 0.82, "data_path": "path/to/data.csv", "run_id": "a1b2c3d4"})
    run_id: str = Field(..., title="ID du run MLflow", description="ID du run MLflow d'entraînement", example="a1b2c3d4e5f6")
    data_path: str = Field(..., title="Chemin des données", description="Chemin vers les données utilisées pour l'entraînement", example="data/processed/processed_data_20250723_120000.csv")
    message: str = Field(..., title="Message", description="Message décrivant le résultat de l'entraînement", example="Modèle entraîné avec succès")
    model_name: str = Field(..., title="Nom du modèle", description="Nom du modèle enregistré", example="dst_trustpilot")
    model_version: str = Field(..., title="Version du modèle", description="Version du modèle enregistré", example="2")
    traceability: Optional[TraceabilityData] = Field(None, title="Traçabilité", description="Informations détaillées sur les ressources effectivement utilisées")

class ValidateRequest(BaseModel):
    """Modèle pour une requête de validation de modèle"""
    model_name: Optional[str] = Field(None, title="Nom du modèle", description="Nom du modèle à valider (optionnel, tous les modèles en attente si non spécifié)", example="dst_trustpilot")
    model_version: Optional[str] = Field(None, title="Version du modèle", description="Version du modèle à valider (obligatoire si model_name est spécifié)", example="2")
    auto_approve: bool = Field(False, title="Approbation automatique", description="Si True, le modèle sera automatiquement promu en production s'il passe la validation", example=False)
    threshold: Optional[float] = Field(None, title="Seuil de validation", description="Seuil d'accuracy pour considérer le modèle comme validé", example=0.75)

class ValidateResponse(BaseModel):
    """Modèle pour une réponse de validation de modèle"""
    status: str = Field(..., title="Statut", description="Statut de la requête de validation", example="success")
    validation_id: str = Field(..., title="ID de validation", description="Identifiant unique de cette session de validation", example="a1b2c3d4")
    models_validated: int = Field(..., title="Nombre de modèles validés", description="Nombre de modèles qui ont été validés", example=1)
    results: List[Dict[str, Any]] = Field(..., title="Résultats", description="Résultats détaillés de la validation pour chaque modèle")
    saved_path: Optional[str] = Field(None, title="Chemin sauvegardé", description="Chemin où sont stockées les données de validation", example="data/validation/validation_results_20250723.csv")

class PromoteResponse(BaseModel):
    """Réponse de promotion d'un modèle"""
    status: str = Field(..., title="Statut", example="success")
    model_name: str = Field(..., title="Nom du modèle")
    model_version: str = Field(..., title="Version du modèle")
    previous_stage: str = Field(..., title="Stage précédent")
    current_stage: str = Field(..., title="Stage actuel")
    message: str = Field(..., title="Message")

class DatasetListResponse(BaseModel):
    """Modèle pour une réponse de liste de jeux de données"""
    datasets: List[Dict[str, Any]]
    total_count: int

class DatasetDetailResponse(BaseModel):
    """Modèle pour une réponse de détails d'un jeu de données, 
    adapté du modèle DatasetInfo de data_api"""
    dataset_id: str
    original_filename: str
    dataset_type: str
    upload_date: str
    rows_count: int
    file_path: str
    run_id: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None

class DistributionStats(BaseModel):
    """Statistiques de distribution des avis"""
    total: int = Field(..., title="Nombre total d'avis")
    positive: int = Field(..., title="Nombre d'avis positifs")
    negative: int = Field(..., title="Nombre d'avis négatifs")
    positive_percent: float = Field(..., title="Pourcentage d'avis positifs")
    negative_percent: float = Field(..., title="Pourcentage d'avis négatifs")

class BalanceRequest(BaseModel):
    """Requête pour équilibrer un jeu de données"""
    dataset_id: str = Field(..., title="ID du jeu de données à équilibrer")
    strategy: str = Field(
        "hybrid", 
        title="Stratégie d'équilibrage",
        description="Méthode utilisée pour équilibrer les données (undersample, oversample, hybrid)",
        example="hybrid"
    )
    target_ratio: float = Field(
        0.5, 
        title="Ratio cible", 
        description="Ratio cible pour la classe minoritaire (avis négatifs) entre 0 et 1",
        example=0.5,
        ge=0.0,
        le=1.0
    )
    random_seed: int = Field(
        42, 
        title="Graine aléatoire", 
        description="Graine pour la reproductibilité",
        example=42
    )

class BalanceResponse(BaseModel):
    """Résultat de l'équilibrage de données"""
    status: str = Field(..., title="Statut", example="success")
    message: str = Field(..., title="Message", example="Jeu de données équilibré avec succès")
    original_dataset_id: str = Field(..., title="ID du jeu de données original")
    balanced_dataset_id: str = Field(..., title="ID du jeu de données équilibré")
    original_distribution: DistributionStats = Field(..., title="Distribution originale")
    balanced_distribution: DistributionStats = Field(..., title="Distribution après équilibrage")
    strategy_used: str = Field(..., title="Stratégie utilisée", example="hybrid")
    target_ratio: float = Field(..., title="Ratio cible visé")
    achieved_ratio: float = Field(..., title="Ratio effectivement atteint")
    execution_time: float = Field(..., title="Temps d'exécution en secondes")
    saved_path: str = Field(..., title="Chemin où les données équilibrées ont été sauvegardées")
    run_id: str = Field(..., title="ID du run MLflow")
    
class ModelInfo(BaseModel):
    """Modèle pour les informations détaillées sur un modèle ML"""
    name: str = Field(..., title="Nom du modèle", description="Nom du modèle enregistré")
    version: str = Field(..., title="Version du modèle", description="Version du modèle enregistré")
    stage: str = Field(..., title="Étape", description="Étape actuelle du modèle (Staging, Production, etc.)")
    creation_timestamp: int = Field(..., title="Timestamp de création", description="Date de création du modèle en timestamp Unix")
    description: Optional[str] = Field(None, title="Description", description="Description du modèle")
    tags: Dict[str, str] = Field(default_factory=dict, title="Tags", description="Tags associés au modèle")
    metrics: Optional[Dict[str, float]] = Field(None, title="Métriques", description="Métriques de performance du modèle")
    
class ModelsResponse(BaseModel):
    """Modèle pour une réponse de liste de modèles"""
    models: List[ModelInfo] = Field(..., title="Liste des modèles")
    production_model: Optional[ModelInfo] = Field(None, title="Modèle actuellement en production")
    pending_models: List[ModelInfo] = Field(default_factory=list, title="Modèles en attente de validation")

class HealthResponse(BaseModel):
    """État de santé de l'API Gateway et des services"""
    status: str = Field(..., title="Statut global", example="ok")
    gateway: Dict[str, Any] = Field(..., title="Informations sur la passerelle")
    services: Dict[str, Dict[str, Any]] = Field(..., title="État des services")
    
# Routes d'authentification

@app.post("/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Obtient un token JWT pour l'authentification.
    
    Utilisez ce token dans l'en-tête Authorization de vos requêtes:
    `Authorization: Bearer {token}`
    """
    # Authentifier l'utilisateur
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning(f"Échec de l'authentification pour l'utilisateur: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Créer un token avec une date d'expiration
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, 
        expires_delta=access_token_expires
    )
    
    logger.info(f"Token généré pour l'utilisateur: {form_data.username}")
    
    # Retourner les informations utilisateur sans le mot de passe
    user_info = {
        "username": user["username"],
        "roles": user["roles"]
    }
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user_info": user_info
    }

@app.get("/me", response_model=UserInfo, tags=["Authentication"])
async def read_users_me(current_user = Depends(get_current_active_user)):
    """
    Retourne les informations de l'utilisateur actuellement authentifié.
    """
    return {
        "username": current_user["username"],
        "roles": current_user["roles"]
    }

# Routes de santé et d'administration

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Vérifie l'état de santé de l'API Gateway et des services.
    """
    services = {
        "prediction": {"status": "unknown"},
        "training": {"status": "unknown"},
        "data": {"status": "unknown"}
    }
    
    # Vérifier l'état de chaque service
    try:
        services["prediction"] = call_service(f"{PREDICTION_API_URL}/health")
    except Exception as e:
        services["prediction"] = {"status": "error", "message": str(e)}
    
    try:
        services["training"] = call_service(f"{TRAINING_API_URL}/health")
    except Exception as e:
        services["training"] = {"status": "error", "message": str(e)}
    
    try:
        services["data"] = call_service(f"{DATA_API_URL}/health")
    except Exception as e:
        services["data"] = {"status": "error", "message": str(e)}
    
    # Déterminer le statut global
    status = "ok"
    for service_name, service_info in services.items():
        if service_info.get("status") != "ok":
            status = "degraded"
            break
    
    return {
        "status": status,
        "gateway": {
            "status": "ok",
            "version": "1.0.0",
            "uptime": "N/A"  # À implémenter si nécessaire
        },
        "services": services
    }

@app.get("/api/docs", tags=["System"])
async def api_docs():
    """
    Redirige vers la documentation de l'API.
    """
    return RedirectResponse(url="/docs")

# Routes pour le service de prédiction

@app.post("/predict", response_model=PredictResponse, description="Prédit le sentiment d'un texte", tags=["Prediction"])
async def predict(
    text: str = Form("Cet article est très bien, je le recommande vivement !", description="Le texte dont on veut prédire le sentiment"),
    model_name: Optional[str] = Form(None, description="Le nom du modèle à utiliser (optionnel)"),
    current_user = Depends(get_current_active_user)
):
    """
    Prédit le sentiment d'un texte donné.
    
    ## Paramètres de formulaire
    - **text**: Le texte dont on veut prédire le sentiment
    - **model_name**: (Optionnel) Le nom du modèle à utiliser
    
    ## Retour
    - Sentiment prédit et score de confiance
    """
    try:
        # Appel au service de prédiction
        data = {"text": text}
        if model_name:
            data["model_name"] = model_name
            
        prediction_result = call_service(f"{PREDICTION_API_URL}/predict", method="POST", data=data)
        
        # Adaptation du format de réponse pour correspondre à PredictResponse
        return {
            "text": text,
            "sentiment": prediction_result["sentiment"],
            "score": max(prediction_result["probabilities"]["négatif"], prediction_result["probabilities"]["positif"]),
            "model_info": {
                "prediction": prediction_result["prediction"],
                "probabilities": prediction_result["probabilities"],
                "model_name": model_name or "dst_trustpilot"  # Nom du modèle par défaut
            }
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service de prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service de prédiction: {str(e)}")

@app.post("/predict/batch", description="Prédit le sentiment de plusieurs textes à partir d'un fichier CSV", tags=["Prediction"])
async def predict_batch(
    file: UploadFile = File(..., description="Fichier CSV avec une colonne 'texte'"),
    model_name: Optional[str] = Form(None, description="Le nom du modèle à utiliser (optionnel)"),
    current_user = Depends(get_current_active_user)
):
    """
    Prédit le sentiment de plusieurs textes à partir d'un fichier CSV.
    
    ## Paramètres de formulaire
    - **file**: Fichier CSV avec une colonne 'texte' (obligatoire)
    - **model_name**: (Optionnel) Le nom du modèle à utiliser
    
    ## Retour
    - Fichier CSV avec les prédictions ajoutées: sentiment, score_positif, score_negatif et prediction
    """
    from fastapi.responses import PlainTextResponse
    
    try:
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
        
        # Envoyer le fichier CSV directement à l'API de prédiction
        file_content = await file.read()
        files = {"file": (file.filename, file_content)}
        data = {}
        
        if model_name:
            data["model_name"] = model_name
            
        # Appel au service de prédiction avec le fichier
        logger.info(f"Envoi du fichier CSV à l'API de prédiction: {file.filename}")
        prediction_results = call_service(
            f"{PREDICTION_API_URL}/predict/batch", 
            method="POST", 
            data=data, 
            files=files
        )
        
        # Vérifier que la réponse contient bien le CSV avec les prédictions
        if "predictions_csv" not in prediction_results:
            logger.error("La réponse de l'API de prédiction ne contient pas de CSV")
            raise HTTPException(
                status_code=500,
                detail="Erreur de format dans la réponse de l'API de prédiction"
            )
        
        # Retourner directement le fichier CSV pour téléchargement en UTF-8 avec séparateur point-virgule
        return PlainTextResponse(
            content=prediction_results["predictions_csv"],
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename=predictions_{file.filename}",
                "Content-Type": "text/csv; charset=utf-8"
            }
        )
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service de prédiction par lots: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service de prédiction par lots: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service de prédiction par lots: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service de prédiction par lots: {str(e)}")

@app.get("/models", description="Liste les modèles disponibles pour la prédiction", tags=["Prediction"])
async def list_models(
    production_only: bool = Query(False, description="Afficher uniquement les modèles en production"),
    current_user = Depends(get_current_active_user)
):
    """
    Liste les modèles disponibles pour la prédiction.
    
    ## Paramètres de requête
    - **production_only**: Si true, n'affiche que les modèles en production (défaut: false)
    
    ## Retour
    - Liste des modèles disponibles pour la prédiction avec leurs versions et statuts
    """
    try:
        # Pour une requête GET, les paramètres doivent être passés dans 'data' et non 'params'
        data = {"production_only": str(production_only).lower()}
        return call_service(f"{PREDICTION_API_URL}/models", data=data)
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service de prédiction (liste des modèles): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service de prédiction: {str(e)}")

# Routes pour le service de données

@app.post("/data/upload", description="Upload de données d'entraînement", tags=["Data"])
async def upload_data(
    file: UploadFile = File(..., description="Fichier CSV contenant les avis clients à traiter pour l'entraînement. Doit contenir les colonnes 'Avis' et 'Note'."),
    current_user = Depends(get_current_active_user)
):
    """
    Upload d'un fichier CSV contenant des données d'entraînement.
    
    ## Paramètres de formulaire
    - **file**: Fichier CSV à uploader (obligatoire). Doit contenir les colonnes 'Avis' et 'Note'.
    
    ## Retour
    - Informations sur le jeu de données traité et stocké
    """
    try:
        # Préparation du fichier à envoyer
        file_content = await file.read()
        files = {"file": (file.filename, file_content)}
        
        # Appel au service de données avec un timeout plus long
        return call_service(f"{DATA_API_URL}/upload", method="POST", files=files, timeout=300)  # 5 minutes
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service de données (upload): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service de données: {str(e)}")

@app.post("/data/upload/validation", description="Upload de données de validation", tags=["Data"])
async def upload_validation_data(
    file: UploadFile = File(..., description="Fichier CSV contenant les avis clients à utiliser comme données de validation. Doit contenir les colonnes 'Avis' et 'Note'."),
    current_user = Depends(get_current_active_user)
):
    """
    Upload d'un fichier CSV contenant des données de validation.
    
    ## Paramètres de formulaire
    - **file**: Fichier CSV à uploader (obligatoire). Doit contenir les colonnes 'Avis' et 'Note'.
    
    ## Retour
    - Informations sur le jeu de données de validation traité et stocké
    """
    try:
        # Préparation du fichier à envoyer
        file_content = await file.read()
        files = {"file": (file.filename, file_content)}
        
        # Appel au service de données avec un timeout plus long
        return call_service(f"{DATA_API_URL}/upload/validation", method="POST", files=files, timeout=300)  # 5 minutes
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service de données (upload validation): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service de données: {str(e)}")

@app.get("/data/datasets", description="Liste les jeux de données disponibles", tags=["Data"])
async def list_datasets(
    current_user = Depends(get_current_active_user)
):
    """
    Liste tous les jeux de données disponibles.
    
    ## Retour
    - Liste des jeux de données avec leurs métadonnées
    """
    try:
        # Appel au service de données sans paramètres
        return call_service(f"{DATA_API_URL}/datasets")
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service de données (liste des datasets): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service de données: {str(e)}")

@app.get("/data/datasets/{dataset_id}", response_model=DatasetDetailResponse, description="Obtient les détails d'un jeu de données", tags=["Data"])
async def get_dataset(
    dataset_id: str = Path(..., title="ID du jeu de données", description="Identifiant unique du jeu de données", example="ab1cd2ef3gh4ij5kl6m"),
    current_user = Depends(get_current_active_user)
):
    """
    Obtient les détails complets d'un jeu de données spécifique.
    
    ## Paramètres
    - **dataset_id**: Identifiant unique du jeu de données
    
    ## Retour
    - Détails complets du jeu de données, incluant métadonnées, statistiques et échantillon de données
    """
    try:
        # Appel au service data_api
        dataset_info = call_service(f"{DATA_API_URL}/datasets/{dataset_id}")
        
        # Adapter le format de DatasetInfo à DatasetDetailResponse
        return {
            "dataset_id": dataset_info["id"],
            "original_filename": dataset_info["name"],
            "dataset_type": dataset_info["type"],
            "upload_date": dataset_info["created_at"],
            "rows_count": dataset_info["n_rows"],
            "file_path": dataset_info["file_path"],
            "run_id": dataset_info.get("run_id"),
            "statistics": dataset_info.get("stats", {})
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service de données (détails dataset): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service de données: {str(e)}")
        
@app.get("/data/datasets/{dataset_id}/download", description="Télécharge un jeu de données au format CSV", tags=["Data"])
async def download_dataset(
    dataset_id: str = Path(..., title="ID du jeu de données", description="Identifiant unique du jeu de données", example="ab1cd2ef3gh4ij5kl6m"),
    current_user = Depends(get_current_active_user)
):
    """
    Télécharge un jeu de données spécifique au format CSV.
    
    ## Paramètres
    - **dataset_id**: Identifiant unique du jeu de données
    
    ## Retour
    - Fichier CSV contenant les données du dataset
    """
    try:
        # Appel au service data_api avec streaming pour éviter de charger tout le fichier en mémoire
        response = await call_service_stream(f"{DATA_API_URL}/datasets/{dataset_id}/download")
        
        # Récupérer les informations sur le fichier depuis les headers
        filename = response.headers.get("content-disposition", "").split("filename=")[-1].strip('"')
        if not filename:
            filename = f"dataset_{dataset_id}.csv"
            
        # Retourner le contenu en streaming
        return StreamingResponse(
            content=response.iter_content(chunk_size=8192),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement du jeu de données: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du téléchargement du jeu de données: {str(e)}")

@app.post("/data/datasets/balance", response_model=BalanceResponse, description="Équilibre un jeu de données pour gérer le déséquilibre des classes", tags=["Data"])
async def balance_dataset(
    dataset_id: str = Form(..., description="ID du jeu de données à équilibrer"),
    strategy: str = Form("hybrid", description="Stratégie d'équilibrage (undersample, oversample, hybrid)"),
    target_ratio: float = Form(0.5, description="Ratio cible pour la classe minoritaire (entre 0 et 1)"),
    random_seed: int = Form(42, description="Graine aléatoire pour la reproductibilité"),
    current_user = Depends(get_current_active_user)
):
    """
    Équilibre un jeu de données existant pour traiter le problème des classes déséquilibrées.
    
    Permet d'améliorer l'entraînement des modèles en présence d'un déséquilibre important
    entre les avis positifs et négatifs (habituellement moins de 1% d'avis négatifs).
    
    ## Paramètres de formulaire
    - **dataset_id**: Identifiant unique du jeu de données à équilibrer
    - **strategy**: Stratégie d'équilibrage à utiliser:
      - *undersample*: Sous-échantillonnage des avis positifs (classe majoritaire)
      - *oversample*: Sur-échantillonnage des avis négatifs (classe minoritaire)
      - *hybrid*: Approche hybride (recommandée) combinant les deux techniques
    - **target_ratio**: Ratio cible pour la classe minoritaire (0-1)
    - **random_seed**: Graine aléatoire pour la reproductibilité
    
    ## Retour
    - Détails sur l'équilibrage réalisé et ID du nouveau jeu de données équilibré
    """
    try:
        # Appel au service data_api pour équilibrer le dataset
        data = {
            "dataset_id": dataset_id,
            "strategy": strategy,
            "target_ratio": target_ratio,
            "random_seed": random_seed
        }
        
        response = call_service(
            f"{DATA_API_URL}/datasets/balance",
            method="POST",
            data=data
        )
        
        # Retourner le résultat de l'équilibrage
        return response
    except Exception as e:
        logger.error(f"Erreur lors de l'équilibrage du jeu de données: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'équilibrage du jeu de données: {str(e)}")

# Routes pour le service d'entraînement

@app.post("/train", response_model=TrainResponse, description="Entraîne un nouveau modèle", tags=["Training"])
async def train_model(
    run_id: Optional[str] = Form(None, description="ID du run MLflow contenant les données d'entraînement (optionnel)", example="ab1cd2ef3gh4ij5kl6m"),
    model_name: Optional[str] = Form("dst_trustpilot", description="Nom à donner au modèle entraîné (optionnel)"),
    base_model_name: Optional[str] = Form(None, description="Nom du modèle à utiliser comme base (optionnel)"),
    base_model_version: Optional[str] = Form(None, description="Version du modèle de base à utiliser (optionnel)"),
    current_user = Depends(get_current_active_user)
):
    """
    Entraîne un nouveau modèle de sentiment sur un jeu de données.
    
    ## Paramètres de formulaire
    - **run_id**: (Optionnel) ID du run MLflow contenant les données d'entraînement.
    - **model_name**: (Optionnel) Nom à donner au modèle entraîné.
    - **base_model_name**: (Optionnel) Nom du modèle à utiliser comme base.
    - **base_model_version**: (Optionnel) Version du modèle de base à utiliser.
    
    ## Retour
    - Informations sur le modèle entraîné et métriques de performance
    """
    try:
        data = {}
        
        if run_id:
            data["run_id"] = run_id
            
        if model_name:
            data["model_name"] = model_name
            
        if base_model_name:
            data["base_model_name"] = base_model_name
            
        if base_model_version:
            data["base_model_version"] = base_model_version
        
        # Timeout augmenté pour permettre à l'entraînement de se terminer
        return call_service(f"{TRAINING_API_URL}/train", method="POST", data=data, timeout=600)  # 10 minutes
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service d'entraînement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service d'entraînement: {str(e)}")

@app.post("/validate", response_model=ValidateResponse, description="Valide un modèle existant", tags=["Training"])
async def validate_model(
    model_name: str = Form("dst_trustpilot", description="Nom du modèle à valider"),
    model_version: str = Form("1", description="Version du modèle à valider"),
    auto_approve: bool = Form(False, description="Approuver le modèle automatiquement si la validation est réussie"),
    threshold: Optional[float] = Form(0.75, description="Seuil d'accuracy pour considérer le modèle comme validé"),
    current_user = Depends(get_current_active_user)
):
    """
    Valide un modèle existant sur le jeu de données de validation.
    
    ## Paramètres de formulaire
    - **model_name**: (Optionnel) Nom du modèle à valider. Si non fourni, tous les modèles en attente seront validés.
    - **model_version**: (Optionnel) Version du modèle à valider. Requis si model_name est spécifié.
    - **auto_approve**: Approuver automatiquement le modèle si la validation est réussie (défaut: False)
    - **threshold**: (Optionnel) Seuil d'accuracy pour considérer le modèle comme validé
    
    ## Retour
    - Métriques de validation et statut de validation du modèle
    """
    try:
        data = {
            "auto_approve": auto_approve
        }
        
        if model_name:
            data["model_name"] = model_name
            
        if model_version:
            data["model_version"] = model_version
            
        if threshold is not None:
            data["threshold"] = threshold
        # Timeout augmenté pour permettre à la validation de se terminer
        return call_service(f"{TRAINING_API_URL}/validate", method="POST", data=data, timeout=300)  # 5 minutes
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service d'entraînement (validation): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service d'entraînement: {str(e)}")

@app.post("/promote/{model_name}/{version}", description="Promeut un modèle en production", tags=["Training"])
async def promote_model(
    model_name: str = Path(..., title="Nom du modèle", description="Nom du modèle à promouvoir", example="dst_trustpilot"),
    version: str = Path(..., title="Version du modèle", description="Version du modèle à promouvoir", example="1"),
    current_user = Depends(get_current_active_user)
):
    """
    Promeut directement un modèle en production sans validation.
    
    ## Paramètres de chemin
    - **model_name**: Nom du modèle à promouvoir
    - **version**: Version du modèle à promouvoir
    
    ## Retour
    - Statut de la promotion et informations sur le modèle promu
    """
    try:
        return call_service(f"{TRAINING_API_URL}/promote/{model_name}/{version}", method="POST")
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service d'entraînement (promotion): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service d'entraînement: {str(e)}")

@app.get("/training/models", response_model=ModelsResponse, description="Liste les modèles entraînés", tags=["Training"])
async def list_training_models(current_user = Depends(get_current_active_user)):
    """
    Liste tous les modèles entraînés avec leurs versions et statuts.
    
    ## Retour
    - Liste des modèles entraînés avec leurs informations (version, statut, métriques)
    - Pour chaque modèle, indique s'il est en production, validé, ou en attente de validation
    """
    try:
        return call_service(f"{TRAINING_API_URL}/models")
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service d'entraînement (liste des modèles): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service d'entraînement: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Le logging est déjà initialisé au début du fichier
    logger.info("Démarrage du serveur API Passerelle")
    
    # Démarrage du serveur
    uvicorn.run(app, host="0.0.0.0", port=8000)
