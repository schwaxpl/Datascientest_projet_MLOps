"""
MLOps Data Ingestion and Training DAG

This DAG:
1. Uploads a CSV file to the Data API for processing
2. Retrieves the run_id from the data upload response
3. Uses the run_id to trigger model training via the Training API
4. Validates the newly trained model
"""

from datetime import datetime, timedelta
import os
import json
import requests
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.hooks.base import BaseHook
from airflow.utils.dates import days_ago

# Default settings
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define URL for Gateway API - can be overridden with Airflow Variables
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

def upload_csv_file(**context):
    """
    Upload a CSV file to the Data API for processing via the Gateway API
    """
    # Get the CSV file path from DAG parameters or context
    csv_file_path = context['dag_run'].conf.get('csv_file_path', '/opt/airflow/dags/data/avis.csv')
    
    print(f"Uploading CSV file: {csv_file_path}")
    
    try:
        # Get authentication headers with token
        headers = get_auth_headers()
        
        # Remove Content-Type from headers for multipart/form-data
        if "Content-Type" in headers:
            del headers["Content-Type"]
        
        # Read the CSV file
        if os.path.exists(csv_file_path):
            with open(csv_file_path, 'rb') as file:
                files = {'file': (os.path.basename(csv_file_path), file, 'text/csv')}
                
                # Make the POST request to upload the file through the Gateway API
                print(f"Sending file to {DATA_API_URL}/upload")
                response = requests.post(
                    f"{DATA_API_URL}/upload",
                    headers=headers,
                    files=files
                )
                
                # Check if the request was successful
                response.raise_for_status()
                
                # Parse the response
                result = response.json()
                print(f"File uploaded successfully. Processed {result.get('n_processed_rows')} rows.")
                
                # Store the run_id and saved_path in XCom for the next task
                context['ti'].xcom_push(key='run_id', value=result.get('run_id'))
                context['ti'].xcom_push(key='saved_path', value=result.get('saved_path'))
                context['ti'].xcom_push(key='stats', value=result.get('stats'))
                
                return result
        else:
            error_msg = f"CSV file not found: {csv_file_path}"
            print(error_msg)
            raise FileNotFoundError(error_msg)
            
    except Exception as e:
        print(f"Error uploading CSV file: {str(e)}")
        raise

def train_model(**context):
    """
    Trigger model training via the Gateway API using the run_id from data upload
    """
    # Get the run_id from the previous task
    run_id = context['ti'].xcom_pull(task_ids='upload_csv_file', key='run_id')
    
    if not run_id:
        error_msg = "No run_id available from previous task"
        print(error_msg)
        raise ValueError(error_msg)
    
    print(f"Training model with run_id: {run_id}")
    
    try:
        # Get authentication headers with token
        headers = get_auth_headers()
        
        # Set up the training request
        training_request = {
            "run_id": run_id,
            "model_name": context['dag_run'].conf.get('model_name', None)  # Optional
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
        
        return result
    
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise

def validate_model(**context):
    """
    Validate the trained model via Gateway API
    """
    # Get the model name and version from the previous task
    model_name = context['ti'].xcom_pull(task_ids='train_model', key='model_name')
    model_version = context['ti'].xcom_pull(task_ids='train_model', key='model_version')
    
    if not model_name or not model_version:
        error_msg = "No model_name or model_version available from previous task"
        print(error_msg)
        raise ValueError(error_msg)
    
    print(f"Validating model: {model_name} v{model_version}")
    
    try:
        # Get authentication headers with token
        headers = get_auth_headers()
        
        # Set up the validation request with auto_approve if specified in the DAG run conf
        auto_approve = context['dag_run'].conf.get('auto_approve', False)
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

# Create the DAG
with DAG(
    'mlops_data_ingestion_training',
    default_args=default_args,
    description='Upload CSV data and train ML model',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'data', 'training'],
) as dag:

    t1 = PythonOperator(
        task_id='upload_csv_file',
        python_callable=upload_csv_file,
        provide_context=True,
    )

    t2 = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
    )

    t3 = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
        provide_context=True,
    )

    t1 >> t2 >> t3
