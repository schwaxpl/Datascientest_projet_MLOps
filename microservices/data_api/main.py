"""
API de gestion des données pour l'analyse de sentiments.
Ce service est responsable de l'ingestion, du traitement et de la gestion des jeux de données.
"""

import os
import time
import uuid
import tempfile
import shutil
from datetime import datetime
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query, Depends, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import io
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager

# Import des modules locaux
from src.data_ingestion import DataIngestionPipeline
from microservices.common.logger_config import init_logging, get_logger
from microservices.common.utils import get_mlflow_client
from microservices.common.config import (
    INGESTION_EXPERIMENT_NAME,
    REQUIRED_COLUMNS
)

# Initialisation du système de logging
loggers = init_logging("data", api=True)
logger = loggers['data']

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Événement exécuté au démarrage de l'application"""
    logger.info("Initialisation de l'API de données...")
    
    # S'assurer que les répertoires de données existent
    os.makedirs('data/processed', exist_ok=True)
    
    yield
    logger.info("Arrêt de l'API de données")

app = FastAPI(
    title="API de Données - Microservice MLOps",
    description="""
    # API de gestion des données pour l'analyse de sentiments
    
    Cette API est responsable de l'ingestion, du traitement et de la gestion des jeux de données.
    
    ## Endpoints
    
    * `/upload` : Upload de fichiers CSV d'avis clients pour l'entraînement
    * `/upload/validation` : Upload de fichiers CSV d'avis clients pour la validation
    * `/datasets` : Liste les jeux de données disponibles
    * `/datasets/{dataset_id}` : Obtient les détails d'un jeu de données spécifique
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
    run_id: Optional[str] = Field(
        None,
        title="ID du run MLflow",
        description="ID du run MLflow contenant les données",
        example="a1b2c3d4e5f6"
    )

class DatasetInfo(BaseModel):
    """Information sur un jeu de données"""
    id: str = Field(..., title="ID du jeu de données")
    name: str = Field(..., title="Nom du jeu de données")
    type: str = Field(..., title="Type de jeu de données", example="entrainement ou validation")
    created_at: str = Field(..., title="Date de création")
    n_rows: int = Field(..., title="Nombre de lignes")
    file_path: str = Field(..., title="Chemin du fichier")
    run_id: Optional[str] = Field(None, title="ID du run MLflow")
    stats: Optional[Dict[str, Any]] = Field(None, title="Statistiques")

class DatasetsResponse(BaseModel):
    """Liste des jeux de données disponibles"""
    datasets: List[DatasetInfo] = Field(..., title="Jeux de données disponibles")
    training_count: int = Field(..., title="Nombre de jeux d'entraînement")
    validation_count: int = Field(..., title="Nombre de jeux de validation")

class HealthResponse(BaseModel):
    """État de santé de l'API"""
    status: str = Field(..., title="Statut", example="ok")
    version: str = Field(..., title="Version de l'API", example="1.0.0")
    mlflow_status: str = Field(..., title="Statut de la connexion MLflow", example="ok")
    disk_usage: Dict[str, Any] = Field(
        ..., 
        title="Utilisation disque", 
        example={
            "total_gb": 100,
            "used_gb": 50,
            "free_gb": 50,
            "percent": 50
        }
    )

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Vérifie l'état de santé de l'API.
    """
    try:
        # Vérifier que le client MLflow est accessible
        client = get_mlflow_client()
        
        # Calculer l'utilisation du disque
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        return {
            "status": "ok",
            "version": "1.0.0",
            "mlflow_status": "ok",
            "disk_usage": {
                "total_gb": total // (2**30),
                "used_gb": used // (2**30),
                "free_gb": free // (2**30),
                "percent": int((used / total) * 100)
            }
        }
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de l'état de santé: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "version": "1.0.0",
                "mlflow_status": "error",
                "disk_usage": {},
                "error": str(e)
            }
        )

@app.get("/datasets", response_model=DatasetsResponse)
def list_datasets():
    """
    Liste tous les jeux de données disponibles, en se basant sur les runs MLflow.
    """
    try:
        datasets = []
        processed_dir = "data/processed"
        
        # Récupérer les runs d'ingestion depuis MLflow
        client = get_mlflow_client()
        experiment_id = client.get_experiment_by_name(INGESTION_EXPERIMENT_NAME).experiment_id
        
        # Récupérer tous les runs de l'expérience d'ingestion
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="status = 'FINISHED'",  # Ne récupérer que les runs réussis
            order_by=["start_time DESC"],
            max_results=1000
        )
        
        logger.info(f"Récupération de {len(runs)} runs MLflow pour l'expérience d'ingestion")
        
        training_count = 0
        validation_count = 0
        
        # Pour chaque run, créer un objet dataset
        for run in runs:
            run_id = run.info.run_id
            
            # Vérifier si le run contient un paramètre data_path
            if "data_path" not in run.data.params:
                logger.warning(f"Run {run_id} ne contient pas de data_path, ignoré")
                continue
                
            data_path = run.data.params["data_path"]
            file_name = os.path.basename(data_path)
            
            # Vérifier si le fichier est un fichier CSV
            if not file_name.endswith(".csv"):
                continue
                
            # Déterminer s'il s'agit d'un jeu de données de validation ou d'entraînement
            is_validation = run.data.params.get("is_validation_set", "false").lower() == "true"
            if not is_validation:
                is_validation = "validation" in file_name.lower()
            
            dataset_type = "validation" if is_validation else "entrainement"
            
            # Essayer de récupérer le fichier local pour obtenir plus d'informations
            file_path = os.path.join(processed_dir, file_name)
            n_rows = 0
            created_at = run.info.start_time / 1000  # MLflow stocke en millisecondes
            
            if os.path.exists(file_path):
                try:
                    # Utiliser la date de modification du fichier
                    stat = os.stat(file_path)
                    created_at = stat.st_mtime
                    
                    # Compter les lignes dans le fichier
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    n_rows = len(df)
                except Exception as e:
                    logger.warning(f"Erreur lors de la lecture du fichier {file_name}: {str(e)}")
            
            # Créer l'objet dataset en utilisant le run_id comme identifiant
            dataset_info = DatasetInfo(
                id=run_id,  # Utiliser directement le run_id comme ID
                name=file_name,
                type=dataset_type,
                created_at=datetime.fromtimestamp(created_at).isoformat(),
                n_rows=n_rows,
                file_path=file_path if os.path.exists(file_path) else data_path,
                run_id=run_id,
                stats=run.data.metrics
            )
            
            datasets.append(dataset_info)
            
            # Incrémenter le compteur approprié
            if is_validation:
                validation_count += 1
            else:
                training_count += 1
        
        return {
            "datasets": sorted(datasets, key=lambda x: x.created_at, reverse=True),
            "training_count": training_count,
            "validation_count": validation_count
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des jeux de données: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des jeux de données: {str(e)}")

@app.get("/datasets/{dataset_id}", response_model=DatasetInfo)
def get_dataset(dataset_id: str):
    """
    Obtient les détails d'un jeu de données spécifique en utilisant directement l'ID du run MLflow.
    """
    try:
        # Récupérer le run MLflow correspondant
        client = get_mlflow_client()
        
        try:
            # Essayer de récupérer directement le run
            run = client.get_run(dataset_id)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du run MLflow {dataset_id}: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Jeu de données avec ID {dataset_id} non trouvé")
        
        # Vérifier que le run appartient à l'expérience d'ingestion
        experiment = client.get_experiment(run.info.experiment_id)
        if experiment.name != INGESTION_EXPERIMENT_NAME:
            logger.warning(f"Le run {dataset_id} n'appartient pas à l'expérience d'ingestion")
            raise HTTPException(status_code=404, detail=f"Jeu de données avec ID {dataset_id} non trouvé")
            
        # Vérifier que le run contient un paramètre data_path
        if "data_path" not in run.data.params:
            logger.warning(f"Le run {dataset_id} ne contient pas de data_path")
            raise HTTPException(status_code=404, detail=f"Jeu de données avec ID {dataset_id} non trouvé")
            
        # Extraire les informations du run
        data_path = run.data.params["data_path"]
        file_name = os.path.basename(data_path)
        
        # Déterminer s'il s'agit d'un jeu de données de validation ou d'entraînement
        is_validation = run.data.params.get("is_validation_set", "false").lower() == "true"
        if not is_validation:
            is_validation = "validation" in file_name.lower()
        
        dataset_type = "validation" if is_validation else "entrainement"
        
        # Essayer de récupérer le fichier local
        processed_dir = "data/processed"
        file_path = os.path.join(processed_dir, file_name)
        n_rows = 0
        created_at = run.info.start_time / 1000
        
        if os.path.exists(file_path):
            try:
                stat = os.stat(file_path)
                created_at = stat.st_mtime
                
                # Compter les lignes
                import pandas as pd
                df = pd.read_csv(file_path)
                n_rows = len(df)
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture du fichier {file_name}: {str(e)}")
        
        # Créer l'objet dataset
        return DatasetInfo(
            id=dataset_id,
            name=file_name,
            type=dataset_type,
            created_at=datetime.fromtimestamp(created_at).isoformat(),
            n_rows=n_rows,
            file_path=file_path if os.path.exists(file_path) else data_path,
            run_id=dataset_id,
            stats=run.data.metrics
        )
    except HTTPException: 
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du jeu de données {dataset_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération du jeu de données: {str(e)}")

@app.get("/datasets/{dataset_id}/download")
def download_dataset(dataset_id: str):
    """
    Télécharge un jeu de données spécifique au format CSV.
    Utilise l'ID du run MLflow pour identifier le dataset et récupère toujours le fichier CSV
    depuis le dossier data_processed des artifacts du run.
    """
    try:
        # Récupérer le run MLflow correspondant
        client = get_mlflow_client()
        
        try:
            # Essayer de récupérer directement le run
            run = client.get_run(dataset_id)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du run MLflow {dataset_id}: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Jeu de données avec ID {dataset_id} non trouvé")
        
        # Vérifier que le run appartient à l'expérience d'ingestion
        experiment = client.get_experiment(run.info.experiment_id)
        if experiment.name != INGESTION_EXPERIMENT_NAME:
            logger.warning(f"Le run {dataset_id} n'appartient pas à l'expérience d'ingestion")
            raise HTTPException(status_code=404, detail=f"Jeu de données avec ID {dataset_id} non trouvé")
        
        # Extraire le nom de fichier depuis les paramètres du run
        data_path = run.data.params.get("data_path", "")
        file_name = os.path.basename(data_path) if data_path else f"dataset_{dataset_id}.csv"
        
        # Récupérer l'artifact depuis MLflow
        try:
            import tempfile
            
            # Créer un répertoire temporaire pour stocker les artifacts
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Téléchargement des artifacts depuis MLflow run {dataset_id}")
            
            # Lister les artifacts du run
            artifacts = client.list_artifacts(run_id=dataset_id)
            
            # Rechercher d'abord dans le dossier data_processed
            data_processed_path = None
            for artifact in artifacts:
                if artifact.is_dir and artifact.path == "data_processed":
                    data_processed_path = artifact.path
                    break
            
            # Si on trouve le dossier data_processed, chercher un fichier CSV dedans
            if data_processed_path:
                data_processed_artifacts = client.list_artifacts(run_id=dataset_id, path=data_processed_path)
                csv_artifacts = [a for a in data_processed_artifacts if a.path.endswith('.csv')]
                
                if csv_artifacts:
                    # Prendre le premier fichier CSV trouvé (ou le seul)
                    csv_path = csv_artifacts[0].path
                    file_name = os.path.basename(csv_path)
                    
                    logger.info(f"Fichier CSV trouvé dans data_processed: {csv_path}")
                    
                    # Télécharger le fichier
                    client.download_artifacts(run_id=dataset_id, path=csv_path, dst_path=temp_dir)
                    downloaded_path = os.path.join(temp_dir, os.path.basename(csv_path))
                    
                    # Remarque: le répertoire temp_dir sera automatiquement nettoyé par le système 
                    # quand le fichier ne sera plus utilisé
                    return FileResponse(
                        path=downloaded_path,
                        filename=file_name,
                        media_type="text/csv"
                    )
            
            # Si on n'a pas trouvé de CSV dans data_processed, chercher dans tous les artifacts
            csv_artifacts = []
            for artifact in artifacts:
                if not artifact.is_dir and artifact.path.endswith('.csv'):
                    csv_artifacts.append(artifact)
            
            if csv_artifacts:
                csv_path = csv_artifacts[0].path
                file_name = os.path.basename(csv_path)
                
                logger.info(f"Fichier CSV trouvé dans les artifacts: {csv_path}")
                
                # Télécharger le fichier
                client.download_artifacts(run_id=dataset_id, path=csv_path, dst_path=temp_dir)
                downloaded_path = os.path.join(temp_dir, os.path.basename(csv_path))
                
                return FileResponse(
                    path=downloaded_path,
                    filename=file_name,
                    media_type="text/csv"
                )
            
            # Si on n'a toujours pas trouvé de CSV, essayer avec le data_path
            if data_path:
                artifact_path = os.path.basename(data_path)
                
                logger.info(f"Tentative de téléchargement avec data_path: {artifact_path}")
                
                # Télécharger le fichier
                client.download_artifacts(run_id=dataset_id, path=artifact_path, dst_path=temp_dir)
                downloaded_path = os.path.join(temp_dir, os.path.basename(artifact_path))
                
                if os.path.exists(downloaded_path):
                    return FileResponse(
                        path=downloaded_path,
                        filename=file_name,
                        media_type="text/csv"
                    )
            
            # Si on n'a toujours rien trouvé
            # Nettoyage manuel du répertoire temporaire car on n'a pas de fichier à renvoyer
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            raise HTTPException(status_code=404, detail="Fichier CSV introuvable dans les artifacts MLflow")
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du fichier depuis MLflow: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération du fichier: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement du dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du téléchargement du dataset: {str(e)}")

@app.post("/upload", response_model=IngestionResponse)
async def upload_data(file: UploadFile = File(
    ..., 
    description="Fichier CSV contenant les avis clients à traiter pour l'entraînement. Doit inclure les colonnes 'Avis' et 'Note'."
)):
    """
    Endpoint pour uploader et traiter un fichier CSV d'avis clients pour l'entraînement.
    
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
                experiment_name=INGESTION_EXPERIMENT_NAME,
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
            
            # Récupérer le run_id du run MLflow créé par le pipeline
            run_id = pipeline.run_id if hasattr(pipeline, 'run_id') else None
            
            response = IngestionResponse(
                status="success",
                n_processed_rows=len(processed_data),
                stats=stats,
                saved_path=output_path,
                run_id=run_id
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

@app.post("/upload/validation", response_model=IngestionResponse)
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
                experiment_name=INGESTION_EXPERIMENT_NAME,
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
            
            # Récupérer le run_id du run MLflow créé par le pipeline
            run_id = pipeline.run_id if hasattr(pipeline, 'run_id') else None
            
            response = IngestionResponse(
                status="success",
                n_processed_rows=len(processed_data),
                stats=stats,
                saved_path=output_path,
                run_id=run_id
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

if __name__ == "__main__":
    import uvicorn
    
    # Le logging est déjà initialisé au début du fichier
    logger.info("Démarrage du serveur API de données")
    
    # Démarrage du serveur
    uvicorn.run(app, host="0.0.0.0", port=8003)
