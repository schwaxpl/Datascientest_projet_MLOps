"""
API d'entraînement pour l'analyse de sentiments.
Ce service est responsable de l'entraînement et de la validation des modèles.
"""

import os
import time
import uuid
import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union
from contextlib import asynccontextmanager

# Import des modules locaux
from microservices.common.logger_config import init_logging, get_logger
from microservices.common.utils import get_mlflow_client, call_service
from microservices.common.config import (
    MODEL_NAME,
    TRAINING_EXPERIMENT_NAME,
    DATA_API_URL
)
from src.train import train_model as train_model_function
from src.model_validation import validate_model, validate_and_promote_model

# Initialisation du système de logging
loggers = init_logging("training", api=True)
logger = loggers['training']

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Événement exécuté au démarrage de l'application"""
    logger.info("Initialisation de l'API d'entraînement...")
    yield
    logger.info("Arrêt de l'API d'entraînement")

app = FastAPI(
    title="API d'Entraînement - Microservice MLOps",
    description="""
    # API d'entraînement et validation de modèles
    
    Cette API est responsable de l'entraînement et de la validation des modèles d'analyse de sentiments.
    
    ## Endpoints
    
    * `/train` : Entraîne un nouveau modèle
    * `/validate` : Valide un modèle existant
    * `/promote/{model_name}/{version}` : Promeut un modèle en production
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
    metrics: Dict[str, Union[float, str]] = Field(
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
    saved_path: Optional[str] = Field(
        None, 
        title="Chemin de sauvegarde", 
        description="Chemin où les données traitées ont été sauvegardées",
        example="data/processed/processed_data_20250723_120000.csv"
    )

class ModelInfo(BaseModel):
    """Information sur un modèle"""
    name: str = Field(..., title="Nom du modèle")
    version: str = Field(..., title="Version du modèle")
    stage: str = Field(..., title="Stage du modèle")
    creation_timestamp: int = Field(..., title="Date de création")
    description: Optional[str] = Field(None, title="Description")
    tags: Optional[Dict[str, str]] = Field(None, title="Tags")
    metrics: Optional[Dict[str, float]] = Field(None, title="Métriques")
    
class ModelsResponse(BaseModel):
    """Liste des modèles disponibles"""
    models: List[ModelInfo] = Field(..., title="Modèles disponibles")
    production_model: Optional[ModelInfo] = Field(None, title="Modèle en production")
    pending_models: List[ModelInfo] = Field([], title="Modèles en attente de validation")

class HealthResponse(BaseModel):
    """État de santé de l'API"""
    status: str = Field(..., title="Statut", example="ok")
    version: str = Field(..., title="Version de l'API", example="1.0.0")
    mlflow_status: str = Field(..., title="Statut de la connexion MLflow", example="ok")

class PromoteResponse(BaseModel):
    """Réponse de promotion d'un modèle"""
    status: str = Field(..., title="Statut", example="success")
    model_name: str = Field(..., title="Nom du modèle")
    model_version: str = Field(..., title="Version du modèle")
    previous_stage: str = Field(..., title="Stage précédent")
    current_stage: str = Field(..., title="Stage actuel")
    message: str = Field(..., title="Message")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Vérifie l'état de santé de l'API.
    """
    try:
        # Vérifier que le client MLflow est accessible
        client = get_mlflow_client()
        _ = client.search_registered_models()
        
        return {
            "status": "ok",
            "version": "1.0.0",
            "mlflow_status": "ok"
        }
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de l'état de santé: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "version": "1.0.0",
                "mlflow_status": "error",
                "error": str(e)
            }
        )

@app.get("/models", response_model=ModelsResponse)
def list_models():
    """
    Liste les modèles disponibles avec leurs informations.
    """
    try:
        client = get_mlflow_client()
        
        # Récupérer la liste des modèles enregistrés
        registered_models = client.search_registered_models()
        
        models = []
        production_model = None
        pending_models = []
        
        for rm in registered_models:
            # Pour chaque modèle enregistré, récupérer les versions
            versions = client.search_model_versions(f"name='{rm.name}'")
            for v in versions:
                # Récupérer les métriques du modèle
                run = client.get_run(v.run_id) if v.run_id else None
                metrics = run.data.metrics if run and run.data else None
                
                # Récupérer les tags du modèle
                tags = {k: v for k, v in v.tags.items() if k != "mlflow.log-model.history"} if hasattr(v, 'tags') else {}
                
                model_info = ModelInfo(
                    name=v.name,
                    version=v.version,
                    stage=v.current_stage,
                    creation_timestamp=v.creation_timestamp,
                    description=v.description,
                    tags=tags,
                    metrics=metrics
                )
                
                models.append(model_info)
                
                # Si le modèle est en production
                if v.current_stage == "Production":
                    production_model = model_info
                
                # Si le modèle est en attente de validation (tag spécifique)
                if "à valider" in tags.values():
                    pending_models.append(model_info)
        
        return {
            "models": models,
            "production_model": production_model,
            "pending_models": pending_models
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des modèles: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des modèles: {str(e)}")

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """
    Entraîne un nouveau modèle d'analyse de sentiments.
    """
    # Création d'un ID unique pour cet entraînement
    train_id = str(uuid.uuid4())[:8]
    logger.info(f"[{train_id}] Nouvelle demande d'entraînement - Run ID: {request.run_id or 'Auto'}, Modèle: {request.model_name or 'Auto'}")
        
    try:
        # Lancement de l'entraînement
        logger.info(f"[{train_id}] Démarrage de l'entraînement{'avec run_id: ' + request.run_id if request.run_id else ''}")
        
        # Mesure du temps d'entraînement
        start_time = time.time()
        
        metrics = train_model_function(
            run_id=request.run_id,
            model_name=request.model_name,
            base_model_name=request.base_model_name,
            base_model_version=request.base_model_version
        )
        
        # Calcul du temps d'entraînement
        execution_time = time.time() - start_time
        logger.info(f"[{train_id}] Entraînement terminé en {execution_time:.3f}s")
        
        # Log des métriques obtenues
        logger.info(f"[{train_id}] Métriques: Train accuracy={metrics['train_accuracy']:.4f}, Test accuracy={metrics['test_accuracy']:.4f}")
        
        # Récupération du run MLflow actuel
        run = mlflow.get_run(run_id=metrics["run_id"])
        if not run:
            logger.warning(f"[{train_id}] Impossible de récupérer les détails du run MLflow: {metrics['run_id']}")
        
        logger.info(f"[{train_id}] Modèle entraîné et enregistré - Run ID: {metrics['run_id']}, Modèle: {metrics['model_name']}, Version: {metrics['model_version']}")
        
        # On restructure les métriques pour les rendre compatibles avec la définition du modèle
        return {
            "status": "success",
            "metrics": {
                "train_accuracy": metrics["train_accuracy"],
                "test_accuracy": metrics["test_accuracy"],
                "data_path": metrics["data_path"],
                "run_id": metrics["run_id"],
                "model_name": metrics["model_name"],
                "model_version": metrics["model_version"]
            },
            "run_id": metrics["run_id"],
            "data_path": metrics.get("data_path", "Unknown"),
            "message": f"Modèle {metrics['model_name']} v{metrics['model_version']} entraîné avec succès (accuracy: {metrics['test_accuracy']:.4f})",
            "model_name": metrics["model_name"],
            "model_version": metrics["model_version"]
        }
    except Exception as e:
        error_message = f"Erreur lors de l'entraînement: {str(e)}"
        logger.error(f"[{train_id}] {error_message}", exc_info=True)
        
        # Ajouter des suggestions pour résoudre les problèmes courants
        suggestions = ""
        if "Error tokenizing data" in str(e) or "Buffer overflow" in str(e):
            suggestions = (
                "\n\nSuggestions de résolution:\n"
                "1. Vérifiez le format du fichier CSV et assurez-vous qu'il est correctement formaté.\n"
                "2. Si les avis contiennent des délimiteurs (virgules, etc.), assurez-vous qu'ils sont correctement échappés.\n"
                "3. Essayez de prétraiter le fichier CSV avec un autre délimiteur comme ';' ou '|'."
            )
        
        raise HTTPException(status_code=500, detail=f"{error_message}{suggestions}")

@app.post("/validate", response_model=ValidationResponse)
async def validate(request: ValidationRequest):
    """
    Valide un modèle existant en utilisant le jeu de données de validation.
    """
    # Création d'un ID unique pour cette validation
    validation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{validation_id}] Nouvelle demande de validation - Modèle: {request.model_name or 'tous'}, Version: {request.model_version or 'dernière'}, Auto-approve: {request.auto_approve}")
    
    try:
        # Appel de la fonction de validation
        start_time = time.time()
        
        if request.model_name and request.model_version:
            # Validation d'un modèle spécifique
            logger.info(f"[{validation_id}] Validation du modèle spécifique: {request.model_name} v{request.model_version}")
            results = validate_and_promote_model(
                model_name=request.model_name,
                version=request.model_version,
                auto_promote=request.auto_approve,
                threshold=request.threshold
            )
            models_validated = 1
        else:
            # Validation de tous les modèles en attente
            logger.info(f"[{validation_id}] Validation de tous les modèles en attente")
            results = validate_model(auto_promote=request.auto_approve, threshold=request.threshold)
            models_validated = len(results)
        
        execution_time = time.time() - start_time
        
        # Log des résultats
        logger.info(f"[{validation_id}] Validation terminée en {execution_time:.3f}s - {models_validated} modèle(s) validé(s)")
        for res in results:
            approved = "approuvé" if res.get("approved", False) else "rejeté"
            logger.info(f"[{validation_id}] Modèle {res['model_name']} v{res['model_version']} {approved} (accuracy: {res.get('accuracy', 0):.4f})")
        
        return {
            "status": "success",
            "validation_id": validation_id,
            "models_validated": models_validated,
            "results": results,
            "saved_path": results[0].get("data_path") if results else None
        }
    except Exception as e:
        logger.error(f"[{validation_id}] Erreur lors de la validation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la validation: {str(e)}")

@app.post("/promote/{model_name}/{version}", response_model=PromoteResponse)
async def promote_model(
    model_name: str = Path(..., title="Nom du modèle à promouvoir"),
    version: str = Path(..., title="Version du modèle à promouvoir")
):
    """
    Promeut directement un modèle en production sans validation.
    """
    try:
        client = get_mlflow_client()
        
        # Vérifier que le modèle et la version existent
        versions = client.search_model_versions(f"name='{model_name}' and version='{version}'")
        if not versions:
            raise HTTPException(status_code=404, detail=f"Modèle {model_name} version {version} non trouvé")
        
        model_version = versions[0]
        previous_stage = model_version.current_stage
        
        # Promouvoir le modèle
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        logger.info(f"Modèle {model_name} version {version} promu en Production (depuis {previous_stage})")
        
        return {
            "status": "success",
            "model_name": model_name,
            "model_version": version,
            "previous_stage": previous_stage,
            "current_stage": "Production",
            "message": f"Modèle {model_name} version {version} promu en Production"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la promotion du modèle {model_name} version {version}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la promotion du modèle: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Le logging est déjà initialisé au début du fichier
    logger.info("Démarrage du serveur API d'entraînement")
    
    # Démarrage du serveur
    uvicorn.run(app, host="0.0.0.0", port=8002)
