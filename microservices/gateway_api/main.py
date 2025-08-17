"""
API Passerelle (Gateway) pour l'architecture microservices MLOps.
Responsable de l'authentification et du routage des requêtes vers les services appropriés.
"""

import os
import time
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Request, status, Form, Path, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from jose import JWTError, jwt
from contextlib import asynccontextmanager

# Import des modules locaux
from microservices.common.logger_config import init_logging, get_logger
from microservices.common.utils import (
    call_service, 
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
    * **Données** : Gestion des jeux de données
    * **Entraînement** : Entraînement et validation de modèles
    * **Administration** : Gestion des utilisateurs et des services
    
    ## Développé dans le cadre du projet MLOps Datascientest.
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
    dataset_id: str = Field(..., title="ID du jeu de données", description="Identifiant du jeu de données à utiliser pour l'entraînement", example="processed_data_20250721_001436.csv")
    model_name: str = Field(..., title="Nom du modèle", description="Nom à donner au modèle entraîné", example="sentiment_model")
    params: Dict[str, Any] = Field(default_factory=dict, title="Paramètres", description="Paramètres d'entraînement du modèle", example={"max_features": 5000, "ngram_range": [1, 2]})

class TrainResponse(BaseModel):
    """Modèle pour une réponse d'entraînement"""
    model_name: str
    model_version: int
    training_info: Dict[str, Any]
    metrics: Dict[str, float]

class ValidateRequest(BaseModel):
    """Modèle pour une requête de validation"""
    model_name: str = Field(..., title="Nom du modèle", description="Nom du modèle à valider", example="sentiment_model")
    version: int = Field(..., title="Version du modèle", description="Version du modèle à valider", example=1)
    validation_dataset: str = Field(..., title="Jeu de données de validation", description="Identifiant du jeu de données à utiliser pour la validation", example="validation.csv")

class ValidateResponse(BaseModel):
    """Modèle pour une réponse de validation"""
    model_name: str
    model_version: int
    validation_info: Dict[str, Any]
    metrics: Dict[str, float]
    is_validated: bool

class DatasetListResponse(BaseModel):
    """Modèle pour une réponse de liste de jeux de données"""
    datasets: List[Dict[str, Any]]
    total_count: int

class DatasetDetailResponse(BaseModel):
    """Modèle pour une réponse de détails d'un jeu de données"""
    dataset_id: str
    original_filename: str
    upload_date: str
    processed_date: str
    rows_count: int
    columns: List[str]
    sample_data: List[Dict[str, Any]]
    statistics: Dict[str, Any]

class HealthResponse(BaseModel):
    """État de santé de l'API Gateway et des services"""
    status: str = Field(..., title="Statut global", example="ok")
    gateway: Dict[str, Any] = Field(..., title="Informations sur la passerelle")
    services: Dict[str, Dict[str, Any]] = Field(..., title="État des services")
    
# Routes d'authentification

@app.post("/token", response_model=Token)
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

@app.get("/me", response_model=UserInfo)
async def read_users_me(current_user = Depends(get_current_active_user)):
    """
    Retourne les informations de l'utilisateur actuellement authentifié.
    """
    return {
        "username": current_user["username"],
        "roles": current_user["roles"]
    }

# Routes de santé et d'administration

@app.get("/health", response_model=HealthResponse)
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

@app.get("/api/docs")
async def api_docs():
    """
    Redirige vers la documentation de l'API.
    """
    return RedirectResponse(url="/docs")

# Routes pour le service de prédiction

@app.post("/predict", response_model=PredictResponse, description="Prédit le sentiment d'un texte")
async def predict(
    text: str = Form(..., description="Le texte dont on veut prédire le sentiment"),
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

@app.post("/predict/batch", description="Prédit le sentiment de plusieurs textes à partir d'un fichier CSV")
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

@app.get("/models", description="Liste les modèles disponibles pour la prédiction")
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

@app.post("/data/upload", description="Upload de données d'entraînement")
async def upload_data(request: Request, current_user = Depends(get_current_active_user)):
    """
    Upload d'un fichier CSV contenant des données d'entraînement.
    
    ## Paramètres de formulaire
    - **file**: Fichier CSV à uploader (obligatoire)
    - **description**: Description optionnelle du jeu de données
    - **tags**: Tags optionnels pour le jeu de données (séparés par des virgules)
    
    ## Retour
    - Informations sur le jeu de données traité et stocké
    """
    try:
        # Pour les requêtes avec fichiers, on doit gérer différemment
        form = await request.form()
        files = {}
        data = {}
        
        for key, value in form.items():
            if hasattr(value, "filename"):
                # C'est un fichier
                file_content = await value.read()
                files[key] = (value.filename, file_content)
            else:
                # C'est un champ de formulaire normal
                data[key] = value
                
        return call_service(f"{DATA_API_URL}/upload", method="POST", data=data, files=files)
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service de données (upload): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service de données: {str(e)}")

@app.post("/data/upload/validation", description="Upload de données de validation")
async def upload_validation_data(request: Request, current_user = Depends(get_current_active_user)):
    """
    Upload d'un fichier CSV contenant des données de validation.
    
    ## Paramètres de formulaire
    - **file**: Fichier CSV à uploader (obligatoire)
    - **description**: Description optionnelle du jeu de données de validation
    - **tags**: Tags optionnels pour le jeu de données (séparés par des virgules)
    
    ## Retour
    - Informations sur le jeu de données de validation traité et stocké
    """
    try:
        # Pour les requêtes avec fichiers, on doit gérer différemment
        form = await request.form()
        files = {}
        data = {}
        
        for key, value in form.items():
            if hasattr(value, "filename"):
                # C'est un fichier
                file_content = await value.read()
                files[key] = (value.filename, file_content)
            else:
                # C'est un champ de formulaire normal
                data[key] = value
                
        return call_service(f"{DATA_API_URL}/upload/validation", method="POST", data=data, files=files)
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service de données (upload validation): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service de données: {str(e)}")

@app.get("/data/datasets", response_model=DatasetListResponse, description="Liste les jeux de données disponibles")
async def list_datasets(
    limit: int = Query(10, ge=1, le=100, description="Nombre maximum de jeux de données à retourner"),
    offset: int = Query(0, ge=0, description="Nombre de jeux de données à sauter"),
    tag: Optional[str] = Query(None, description="Filtrer par tag"),
    current_user = Depends(get_current_active_user)
):
    """
    Liste tous les jeux de données disponibles avec pagination et filtrage.
    
    ## Paramètres de requête
    - **limit**: Nombre maximum de jeux de données à retourner (défaut: 10, max: 100)
    - **offset**: Nombre de jeux de données à sauter (défaut: 0)
    - **tag**: Filtrer par tag (optionnel)
    
    ## Retour
    - Liste des jeux de données avec leurs métadonnées
    """
    try:
        params = {"limit": limit, "offset": offset}
        if tag:
            params["tag"] = tag
        return call_service(f"{DATA_API_URL}/datasets", params=params)
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service de données (liste des datasets): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service de données: {str(e)}")

@app.get("/data/datasets/{dataset_id}", response_model=DatasetDetailResponse, description="Obtient les détails d'un jeu de données")
async def get_dataset(
    dataset_id: str = Path(..., title="ID du jeu de données", description="Identifiant unique du jeu de données", example="processed_data_20250721_001436.csv"),
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
        return call_service(f"{DATA_API_URL}/datasets/{dataset_id}")
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service de données (détails dataset): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service de données: {str(e)}")

# Routes pour le service d'entraînement

@app.post("/train", response_model=TrainResponse, description="Entraîne un nouveau modèle")
async def train_model(
    dataset_id: str = Form(..., description="Identifiant du jeu de données à utiliser"),
    model_name: str = Form(..., description="Nom à donner au modèle entraîné"),
    max_features: Optional[int] = Form(None, description="Nombre maximum de features à utiliser"),
    ngram_range_min: Optional[int] = Form(1, description="Valeur minimale pour n-gram"),
    ngram_range_max: Optional[int] = Form(2, description="Valeur maximale pour n-gram"),
    current_user = Depends(get_current_active_user)
):
    """
    Entraîne un nouveau modèle de sentiment sur un jeu de données.
    
    ## Paramètres de formulaire
    - **dataset_id**: Identifiant du jeu de données à utiliser pour l'entraînement
    - **model_name**: Nom à donner au modèle entraîné
    - **max_features**: (Optionnel) Nombre maximum de features à utiliser
    - **ngram_range_min**: (Optionnel) Valeur minimale pour n-gram (défaut: 1)
    - **ngram_range_max**: (Optionnel) Valeur maximale pour n-gram (défaut: 2)
    
    ## Retour
    - Informations sur le modèle entraîné et métriques de performance
    """
    try:
        data = {
            "dataset_id": dataset_id,
            "model_name": model_name,
            "params": {}
        }
        
        if max_features:
            data["params"]["max_features"] = max_features
        
        data["params"]["ngram_range"] = [ngram_range_min, ngram_range_max]
        
        return call_service(f"{TRAINING_API_URL}/train", method="POST", data=data)
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service d'entraînement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service d'entraînement: {str(e)}")

@app.post("/validate", response_model=ValidateResponse, description="Valide un modèle existant")
async def validate_model(
    model_name: str = Form(..., description="Nom du modèle à valider"),
    version: str = Form(..., description="Version du modèle à valider"),
    validation_dataset: str = Form(..., description="Identifiant du jeu de données de validation"),
    approve: bool = Form(False, description="Approuver le modèle automatiquement si la validation est réussie"),
    current_user = Depends(get_current_active_user)
):
    """
    Valide un modèle existant sur un jeu de données de validation.
    
    ## Paramètres de formulaire
    - **model_name**: Nom du modèle à valider
    - **version**: Version du modèle à valider
    - **validation_dataset**: Identifiant du jeu de données de validation à utiliser
    - **approve**: Approuver automatiquement le modèle si la validation est réussie (défaut: False)
    
    ## Retour
    - Métriques de validation et statut de validation du modèle
    """
    try:
        data = {
            "model_name": model_name,
            "version": version,
            "validation_dataset": validation_dataset,
            "approve": approve
        }
        return call_service(f"{TRAINING_API_URL}/validate", method="POST", data=data)
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au service d'entraînement (validation): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au service d'entraînement: {str(e)}")

@app.post("/promote", description="Promeut un modèle en production")
async def promote_model(
    model_name: str = Form(..., description="Nom du modèle à promouvoir"),
    version: str = Form(..., description="Version du modèle à promouvoir"),
    current_user = Depends(get_current_active_user)
):
    """
    Promeut un modèle validé en production.
    
    ## Paramètres de formulaire
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

@app.get("/training/models", description="Liste les modèles entraînés")
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
