"""
API REST FastAPI pour le service de prédiction.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import numpy as np
from typing import List, Dict, Optional
import tempfile
import os
import json
from datetime import datetime
import mlflow
import mlflow.keras
import pickle
from src.predict import PredictionService
from src.data_ingestion import DataIngestionPipeline
from contextlib import asynccontextmanager
from tensorflow.keras.models import load_model
app = FastAPI(
    title="MLOps Text Classification API",
    description="API pour la classification de texte avec modèle TF-IDF",
    version="1.0.0"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Événement exécuté au démarrage de l'application"""
    print("Initialisation de l'application...")
    init_mlflow_model()
    print("Initialisation terminée")
    yield

app = FastAPI(
    title="MLOps Text Classification API",
    description="API pour la classification de texte avec modèle TF-IDF",
    version="1.0.0",
    lifespan=lifespan
)


from src.config import (
    MODEL_NAME,
    MODEL_PATH,
    VECTORIZER_PATH,
    MLFLOW_TRACKING_URI
)
from src.utils import get_latest_model_version, load_model_from_registry, get_vectorizer_from_run

# Initialisation lazy du service de prédiction et du vectorizer
prediction_service = None
vectorizer = None

def load_vectorizer():
    """Charge le vectorizer depuis le fichier local"""
    global vectorizer
    if vectorizer is None:
        print(f"Chargement du vectorizer depuis {VECTORIZER_PATH}")
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
    return vectorizer

def init_mlflow_model():
    """
    Initialise le modèle dans MLflow Model Registry si nécessaire.
    Crée une première version à partir du modèle local si aucune version n'existe.
    """
    # Configuration de l'URI de tracking MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Création du client MLflow
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Vérifier si le modèle existe dans le registre
        try:
            model = client.get_registered_model(MODEL_NAME)
        except:
            print(f"Création du modèle {MODEL_NAME} dans MLflow Model Registry")
            client.create_registered_model(MODEL_NAME)
        
        # Vérifier s'il existe des versions
        try:
            get_latest_model_version(client, MODEL_NAME)
        except ValueError:
            print("Aucune version trouvée. Création de la version initiale...")
            
            # Créer un run MLflow pour enregistrer le modèle initial
            with mlflow.start_run(run_name="initial_model_registration"):
                try:
                    # Charger le modèle directement comme un modèle Keras

                    model = load_model(MODEL_PATH)
                except Exception as e:
                    print(f"Erreur lors du chargement du modèle comme modèle Keras: {str(e)}")
                    print("Tentative de chargement comme modèle pickle...")
                    with open(MODEL_PATH, 'rb') as f:
                        model = pickle.load(f)
                
                # Charger le vectorizer
                with open(VECTORIZER_PATH, 'rb') as f:
                    vectorizer = pickle.load(f)
                
                # Log du vectorizer comme artifact
                mlflow.log_artifact(VECTORIZER_PATH)
                
                # Log du modèle avec MLflow, en spécifiant explicitement keras comme flavor
                mlflow.tensorflow.log_model(
                    tf_model = model,
                    artifact_path="model",
                    registered_model_name=MODEL_NAME
                )
            print("Version initiale créée avec succès")
            
    except Exception as e:
        print(f"Erreur lors de l'initialisation du modèle dans MLflow: {str(e)}")
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
        # Configuration de l'URI de tracking MLflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Chargement du modèle depuis MLflow
        client = mlflow.tracking.MlflowClient()
        model = load_model_from_registry(model_name or MODEL_NAME, version=model_version)
        
        # Utilisation du vectorizer local
        vectorizer = load_vectorizer()
        
        return PredictionService.from_artifacts(
            model=model,
            vectorizer=vectorizer
        )
        
    # Utiliser le cache si aucun modèle spécifique n'est demandé
    if prediction_service is None:
        # Configuration de l'URI de tracking MLflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Récupération de la dernière version du modèle
        client = mlflow.tracking.MlflowClient()
        latest_version = get_latest_model_version(client, MODEL_NAME)
        print(f"Chargement du modèle version: {latest_version.version}")
        
        # Chargement du modèle depuis MLflow
        model = load_model_from_registry(MODEL_NAME)
        
        # Utilisation du vectorizer local
        vectorizer = load_vectorizer()
        
        # Création du service avec le modèle et le vectorizer
        prediction_service = PredictionService.from_artifacts(
            model=model,
            vectorizer=vectorizer
        )
        
    return prediction_service

class PredictionRequest(BaseModel):
    text: str
    model_name: Optional[str] = None  # Nom du modèle à utiliser pour la prédiction
    model_version: Optional[str] = None  # Version spécifique du modèle à utiliser

class PredictionResponse(BaseModel):
    prediction: int  # 0 pour négatif, 1 pour positif
    probabilities: Dict[str, float]  # Probabilités pour chaque classe
    sentiment: str  # "négatif" ou "positif" en texte

class TrainingRequest(BaseModel):
    run_id: Optional[str] = None  # ID du run MLflow contenant les données d'entraînement
    model_name: Optional[str] = None  # Nom sous lequel enregistrer le nouveau modèle
    base_model_name: Optional[str] = None  # Nom du modèle à utiliser comme base
    base_model_version: Optional[str] = None  # Version du modèle de base à utiliser

class TrainingResponse(BaseModel):
    status: str
    metrics: Dict[str, float]
    run_id: str  # ID du run MLflow d'entraînement
    data_path: str
    message: str
    model_name: str  # Nom du modèle enregistré
    model_version: str  # Version du modèle enregistré

class IngestionResponse(BaseModel):
    status: str
    n_processed_rows: int
    stats: Dict
    saved_path: str

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    Endpoint pour uploader et traiter un fichier CSV d'avis clients.
    Le fichier doit contenir au minimum les colonnes 'Avis' et 'Note'.
    """
    try:
        # Vérification de l'extension
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Le fichier doit être au format CSV")
        
        # Création d'un dossier temporaire pour stocker le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            # Création du pipeline d'ingestion
            pipeline = DataIngestionPipeline(
                data_path=tmp_file.name,
                experiment_name="data_ingestion_api"
            )
            
            # Exécution du pipeline
            processed_data = pipeline.run_pipeline()
            
            # Création du dossier processed s'il n'existe pas
            os.makedirs('data/processed', exist_ok=True)
            
            # Sauvegarde des données traitées
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/processed/processed_data_{timestamp}.csv"
            processed_data.to_csv(output_path, index=False)
            
            # Calcul des statistiques
            stats = pipeline.get_data_stats(processed_data)
            
            return IngestionResponse(
                status="success",
                n_processed_rows=len(processed_data),
                stats=stats,
                saved_path=output_path
            )
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Nettoyage du fichier temporaire
        if 'tmp_file' in locals():
            os.unlink(tmp_file.name)

@app.get("/")
def read_root():
    """Point d'entrée de l'API"""
    return {"message": "Bienvenue sur l'API de prédiction"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Endpoint pour la classification de texte
    
    Args:
        request (PredictionRequest): Requête contenant le texte à classifier
        
    Returns:
        PredictionResponse: Prédiction du modèle avec les détails suivants :
            - prediction: 0 pour négatif, 1 pour positif
            - probabilities: probabilités pour chaque classe
            - sentiment: "négatif" ou "positif" en texte
    """
    try:
        # Conversion du texte en série pandas
        text_series = pd.Series([request.text])
        
        # Prédiction avec le modèle spécifié
        service = get_prediction_service(
            model_name=request.model_name,
            model_version=request.model_version
        )
        prediction_proba = service.predict_proba(text_series)
        
        # Le modèle retourne un tableau de forme (1, 2) avec des probabilités
        if not isinstance(prediction_proba, np.ndarray) or prediction_proba.ndim != 2:
            raise ValueError("Format de prédiction invalide")
            
        # Extraire les probabilités
        neg_proba, pos_proba = prediction_proba[0]
        
        # Déterminer la classe prédite
        predicted_class = 1 if pos_proba > neg_proba else 0
        sentiment = "positif" if predicted_class == 1 else "négatif"
        
        return PredictionResponse(
            prediction=predicted_class,
            probabilities={
                "négatif": float(neg_proba),
                "positif": float(pos_proba)
            },
            sentiment=sentiment
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """
    Endpoint pour entraîner le modèle sur de nouvelles données.
    Utilise les données d'un run MLflow d'ingestion spécifique ou le dernier run réussi.
    
    Args:
        request (TrainingRequest): Requête contenant optionnellement l'ID du run MLflow
        
    Returns:
        TrainingResponse: Résultat de l'entraînement avec les métriques et informations
    """
    try:
        # Import local pour éviter les problèmes de dépendances circulaires
        from src.train import train_model as train
        
        # Lancement de l'entraînement
        print(f"Démarrage de l'entraînement{'avec run_id: ' + request.run_id if request.run_id else ''}")
        
        metrics = train(
            run_id=request.run_id,
            model_name=request.model_name,
            base_model_name=request.base_model_name,
            base_model_version=request.base_model_version
        )
        print("Entraînement terminé avec succès")
        # Récupération du run MLflow actuel
        run = mlflow.get_run(run_id=metrics["run_id"])
        if not run:
            raise ValueError("Impossible de récupérer le run MLflow")
        print(f"Run ID: {run.info.run_id}")
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
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'entraînement: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
