"""
MLOps Automated Data Ingestion and Training DAG

This DAG:
1. Monitors MLflow for new datasets uploaded in the last minute
2. Automatically triggers model training if new data is detected
3. Validates the newly trained model
4. Runs every minute to ensure rapid response to new data

The DAG checks for new datasets in MLflow artifacts and processes them automatically,
enabling a fully automated ML pipeline triggered by data uploads.
"""

from datetime import datetime, timedelta
import os
import json
import requests
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.hooks.base import BaseHook
from airflow.utils.dates import days_ago
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults

# Default settings for the automated DAG
default_args = {
    'owner': 'mlops_automation',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
}

# MLflow configuration
try:
    MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")
except KeyError:
    MLFLOW_TRACKING_URI = "http://mlflow:5000"

# Define URL for Gateway API
try:
    GATEWAY_API_URL = Variable.get("GATEWAY_API_URL")
except KeyError:
    GATEWAY_API_URL = "http://gateway-api:8000"

# All API endpoints are accessed through the Gateway
DATA_API_URL = f"{GATEWAY_API_URL}/data"
TRAINING_API_URL = f"{GATEWAY_API_URL}/training"
PREDICTION_API_URL = f"{GATEWAY_API_URL}/prediction"

# Define credentials - should be stored in Airflow connections
def get_auth_headers():
    """Get authentication headers using username and password from Airflow connections"""
    try:
        # Try to get credentials from Airflow connection
        conn = BaseHook.get_connection("gateway_api")
        username = conn.login
        password = conn.password
        
        # Use environment variables if available as backup
        if not username or not password:
            username = os.environ.get("ADMIN_USERNAME", "admin")
            password = os.environ.get("ADMIN_PASSWORD", "adminpassword")
            print("Using environment variables for API authentication")
        
        # Get JWT token from Gateway API
        auth_response = requests.post(
            f"{GATEWAY_API_URL}/token",
            data={"username": username, "password": password}
        )
        auth_response.raise_for_status()
        token = auth_response.json().get("access_token")
        
        if not token:
            raise ValueError("No token received from authentication endpoint")
            
        print("Successfully obtained authentication token")
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    except Exception as e:
        print(f"Error getting auth headers: {str(e)}")
        # Fallback to default credentials (not secure, for development only)
        try:
            auth_response = requests.post(
                f"{GATEWAY_API_URL}/token",
                data={"username": "admin", "password": "adminpassword"}
            )
            auth_response.raise_for_status()
            token = auth_response.json().get("access_token")
            if not token:
                raise ValueError("No token received from authentication endpoint using default credentials")
            print("Using default credentials for API authentication")
            return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        except Exception as inner_e:
            print(f"Critical authentication error: {str(inner_e)}")
            raise

def check_new_datasets(**context):
    """
    Vérifie s'il y a de nouveaux datasets uploadés dans MLflow dans la dernière minute
    """
    try:
        # Configurer MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        # Calculer le timestamp d'il y a une minute
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        timestamp_ms = int(one_minute_ago.timestamp() * 1000)
        
        print(f"Checking for new datasets since: {one_minute_ago}")
        print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
        
        # Chercher les expériences récentes
        try:
            experiments = client.search_experiments()
            print(f"Found {len(experiments)} experiments to check")
        except Exception as e:
            print(f"Error accessing MLflow experiments: {str(e)}")
            print("MLflow server may not be accessible")
            context['ti'].xcom_push(key='has_new_data', value=False)
            return False
            
        new_datasets = []
        
        for experiment in experiments:
            try:
                # Chercher les runs récents dans cette expérience
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"attribute.start_time >= {timestamp_ms}",
                    max_results=50
                )
                
                for run in runs:
                    # Vérifier s'il y a des artifacts de type dataset
                    try:
                        artifacts = client.list_artifacts(run.info.run_id)
                    except Exception as artifact_error:
                        print(f"Error accessing artifacts for run {run.info.run_id}: {str(artifact_error)}")
                        continue
                    
                    for artifact in artifacts:
                        # Chercher les fichiers CSV ou datasets
                        if (artifact.path.endswith('.csv') or 
                            'dataset' in artifact.path.lower() or
                            'data' in artifact.path.lower()):
                            
                            dataset_info = {
                                "run_id": run.info.run_id,
                                "experiment_id": experiment.experiment_id,
                                "experiment_name": experiment.name,
                                "artifact_path": artifact.path,
                                "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                                "run_name": run.data.tags.get("mlflow.runName", f"run_{run.info.run_id[:8]}")
                            }
                            
                            new_datasets.append(dataset_info)
                            print(f"Found new dataset: {artifact.path} in run {run.info.run_id}")
            
            except Exception as e:
                print(f"Error checking experiment {experiment.name}: {str(e)}")
                continue
        
        if new_datasets:
            print(f"Found {len(new_datasets)} new datasets!")
            # Stocker les informations pour les tâches suivantes
            context['ti'].xcom_push(key='new_datasets', value=new_datasets)
            context['ti'].xcom_push(key='has_new_data', value=True)
            
            # Prendre le dataset le plus récent pour le traitement
            latest_dataset = max(new_datasets, key=lambda x: x['start_time'])
            context['ti'].xcom_push(key='selected_dataset', value=latest_dataset)
            
            return True
        else:
            print("No new datasets found in the last minute")
            context['ti'].xcom_push(key='has_new_data', value=False)
            return False
            
    except Exception as e:
        print(f"Error checking for new datasets: {str(e)}")
        context['ti'].xcom_push(key='has_new_data', value=False)
        return False

def train_model_from_mlflow(**context):
    """
    Lance l'entraînement directement avec le run_id MLflow détecté
    """
    # Vérifier s'il y a des nouvelles données
    has_new_data = context['ti'].xcom_pull(task_ids='check_new_datasets', key='has_new_data')
    
    if not has_new_data:
        print("No new data detected, skipping training")
        return None
    
    # Récupérer les informations du dataset MLflow sélectionné
    selected_dataset = context['ti'].xcom_pull(task_ids='check_new_datasets', key='selected_dataset')
    
    if not selected_dataset:
        raise ValueError("No dataset selected for training")
    
    mlflow_run_id = selected_dataset['run_id']
    experiment_name = selected_dataset.get('experiment_name', 'unknown')
    
    # Générer un nom de modèle basé sur les métadonnées MLflow
    source_run_id = mlflow_run_id[:8]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_name = f"auto_{experiment_name}_{source_run_id}_{timestamp}".replace(' ', '_').lower()
    
    print(f"Training model '{model_name}' with MLflow run_id: {mlflow_run_id}")
    print(f"Source MLflow experiment: {experiment_name}")
    
    try:
        # Get authentication headers with token
        headers = get_auth_headers()
        
        # Set up the training request - utilise directement le run_id MLflow
        training_request = {
            "run_id": mlflow_run_id,  # Utilise le run_id MLflow directement
            "model_name": model_name
        }
        
        # Make the POST request to train the model through the Gateway API
        print(f"Sending training request to {TRAINING_API_URL}/train")
        response = requests.post(
            f"{TRAINING_API_URL}/train",
            headers=headers,
            json=training_request
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        print(f"Model trained successfully. Model: {result.get('model_name')} v{result.get('model_version')}")
        
        # Store model info in XCom for the next task
        context['ti'].xcom_push(key='model_name', value=result.get('model_name'))
        context['ti'].xcom_push(key='model_version', value=result.get('model_version'))
        context['ti'].xcom_push(key='metrics', value=result.get('metrics'))
        context['ti'].xcom_push(key='mlflow_source_run_id', value=mlflow_run_id)
        
        return result
    
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise
        context['ti'].xcom_push(key='model_name', value=result.get('model_name'))
        context['ti'].xcom_push(key='model_version', value=result.get('model_version'))
        context['ti'].xcom_push(key='metrics', value=result.get('metrics'))
        
        return result
    
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise

def validate_model(**context):
    """
    Validate the trained model via Gateway API
    Auto-approve les modèles issus de nouveaux datasets MLflow
    """
    # Vérifier s'il y a des nouvelles données
    has_new_data = context['ti'].xcom_pull(task_ids='check_new_datasets', key='has_new_data')
    
    if not has_new_data:
        print("No new data detected, skipping validation")
        return None
    
    # Get the model name and version from the training task
    model_name = context['ti'].xcom_pull(task_ids='train_model_from_mlflow', key='model_name')
    model_version = context['ti'].xcom_pull(task_ids='train_model_from_mlflow', key='model_version')
    
    if not model_name or not model_version:
        error_msg = "No model_name or model_version available from training task"
        print(error_msg)
        raise ValueError(error_msg)
    
    print(f"Validating model: {model_name} v{model_version}")
    
    try:
        # Get authentication headers with token
        headers = get_auth_headers()
        
        # Auto-approve pour les modèles issus de MLflow
        auto_approve = True  # Validation automatique pour les pipelines automatisés
        validation_request = {
            "model_name": model_name,
            "model_version": model_version,
            "auto_approve": auto_approve
        }
        
        # Make the POST request to validate the model through the Gateway API
        print(f"Sending validation request to {TRAINING_API_URL}/validate")
        response = requests.post(
            f"{TRAINING_API_URL}/validate",
            headers=headers,
            json=validation_request
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        print(f"Model validation completed. Results: {json.dumps(result.get('results'), indent=2)}")
        
        return result
    
    except Exception as e:
        print(f"Error validating model: {str(e)}")
        raise

def cleanup_downloaded_files(**context):
    """
    Pas de nettoyage nécessaire car on ne télécharge plus de fichiers
    """
    print("No file downloads in this pipeline, no cleanup needed")
    return

def log_pipeline_summary(**context):
    """
    Log un résumé détaillé du pipeline automatisé incluant les métadonnées MLflow
    """
    has_new_data = context['ti'].xcom_pull(task_ids='check_new_datasets', key='has_new_data')
    
    if not has_new_data:
        print("No new data processed in this execution")
        return {"status": "no_new_data", "execution_date": context['execution_date'].isoformat()}
    
    # Collect information from all tasks
    new_datasets = context['ti'].xcom_pull(task_ids='check_new_datasets', key='new_datasets')
    selected_dataset = context['ti'].xcom_pull(task_ids='check_new_datasets', key='selected_dataset')
    model_name = context['ti'].xcom_pull(task_ids='train_model_from_mlflow', key='model_name')
    model_version = context['ti'].xcom_pull(task_ids='train_model_from_mlflow', key='model_version')
    metrics = context['ti'].xcom_pull(task_ids='train_model_from_mlflow', key='metrics')
    mlflow_source_run_id = context['ti'].xcom_pull(task_ids='train_model_from_mlflow', key='mlflow_source_run_id')
    
    summary = {
        "dag_run_id": context['dag_run'].run_id,
        "execution_date": context['execution_date'].isoformat(),
        "pipeline_type": "automated_mlflow_trigger",
        "trigger_info": {
            "total_new_datasets": len(new_datasets) if new_datasets else 0,
            "selected_dataset": selected_dataset,
            "mlflow_source_run": selected_dataset.get('run_id') if selected_dataset else None,
            "mlflow_experiment": selected_dataset.get('experiment_name') if selected_dataset else None
        },
        "processing_results": {
            "mlflow_source_run_id": mlflow_source_run_id,
            "model_name": model_name,
            "model_version": model_version,
            "metrics": metrics
        },
        "status": "completed_successfully",
        "note": "Training was done directly from MLflow run_id without file download"
    }
    
    print("=== AUTOMATED PIPELINE EXECUTION SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    
    return summary

# Create the automated DAG
with DAG(
    'mlops_automated_training',
    default_args=default_args,
    description='Automated ML pipeline triggered by new MLflow datasets',
    schedule_interval=timedelta(minutes=1),  # Exécute toutes les minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'automated', 'mlflow', 'training'],
    max_active_runs=1,  # Éviter les exécutions concurrentes
) as dag:

    # Task 1: Check for new datasets in MLflow
    check_datasets_task = PythonOperator(
        task_id='check_new_datasets',
        python_callable=check_new_datasets,
        provide_context=True,
    )

    # Task 2: Train model directly from MLflow run_id (conditional)
    train_task = PythonOperator(
        task_id='train_model_from_mlflow',
        python_callable=train_model_from_mlflow,
        provide_context=True,
    )

    # Task 3: Validate model (conditional)
    validate_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
        provide_context=True,
    )

    # Task 4: Log pipeline summary
    summary_task = PythonOperator(
        task_id='log_pipeline_summary',
        python_callable=log_pipeline_summary,
        provide_context=True,
    )

    # Task 5: Cleanup (dummy task since no files are downloaded)
    cleanup_task = PythonOperator(
        task_id='cleanup_downloaded_files',
        python_callable=cleanup_downloaded_files,
        provide_context=True,
    )

    # Define task dependencies - simplified workflow
    check_datasets_task >> train_task >> validate_task >> [summary_task, cleanup_task]
