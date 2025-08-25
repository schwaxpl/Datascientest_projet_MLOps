"""
MLOps Scheduled Data Monitoring DAG

This DAG:
1. Monitors a designated directory for new CSV files
2. For each new file, processes it through the Data API
3. Optionally triggers model training on new data
4. Logs the status of all operations
"""

from datetime import datetime, timedelta
import os
import json
import glob
import requests
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

# Directory to monitor for new files
try:
    DATA_DIR = Variable.get("DATA_MONITOR_DIR")
except KeyError:
    DATA_DIR = "/opt/airflow/dags/data"
# Directory to move processed files to
try:
    PROCESSED_DIR = Variable.get("DATA_PROCESSED_DIR")
except KeyError:
    PROCESSED_DIR = "/opt/airflow/dags/data/processed"
# Whether to trigger training automatically
try:
    AUTO_TRAIN = Variable.get("AUTO_TRAIN").lower() == "true"
except KeyError:
    AUTO_TRAIN = False

# Define credentials retrieval - same as in previous DAG
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

def find_new_csv_files(**context):
    """Find new CSV files in the monitored directory"""
    # Ensure the directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files in {DATA_DIR}")
    
    # Push the list of files to XCom
    context['ti'].xcom_push(key='csv_files', value=csv_files)
    
    return csv_files

def process_csv_files(**context):
    """Process each new CSV file found via Gateway API"""
    # Get the list of files from XCom
    csv_files = context['ti'].xcom_pull(task_ids='find_new_csv_files', key='csv_files')
    
    if not csv_files:
        print("No CSV files to process")
        return []
    
    processed_files = []
    results = []
    
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        
        try:
            # Get authentication headers with token
            headers = get_auth_headers()
            
            # Remove Content-Type from headers for multipart/form-data
            if "Content-Type" in headers:
                del headers["Content-Type"]
            
            with open(csv_file, 'rb') as file:
                files = {'file': (os.path.basename(csv_file), file, 'text/csv')}
                
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
                
                # Add to results
                result['source_file'] = csv_file
                results.append(result)
                processed_files.append({
                    'file': csv_file,
                    'run_id': result.get('run_id'),
                    'saved_path': result.get('saved_path')
                })
                
                print(f"Successfully processed {csv_file}")
                
                # Move the file to the processed directory
                processed_file = os.path.join(PROCESSED_DIR, os.path.basename(csv_file))
                os.rename(csv_file, processed_file)
                print(f"Moved {csv_file} to {processed_file}")
                
        except Exception as e:
            print(f"Error processing file {csv_file}: {str(e)}")
    
    # Push the results to XCom
    context['ti'].xcom_push(key='processed_files', value=processed_files)
    context['ti'].xcom_push(key='process_results', value=results)
    
    return processed_files

def trigger_training_if_needed(**context):
    """Trigger model training if enabled via Gateway API"""
    if not AUTO_TRAIN:
        print("Automatic training is disabled. Skipping.")
        return None
    
    # Get the processed files from XCom
    processed_files = context['ti'].xcom_pull(task_ids='process_csv_files', key='processed_files')
    
    if not processed_files:
        print("No files were processed. Skipping training.")
        return None
    
    # Get the latest processed file's run_id
    latest_file = processed_files[-1]
    run_id = latest_file.get('run_id')
    
    if not run_id:
        print("No run_id available for the latest processed file. Skipping training.")
        return None
    
    print(f"Triggering training with run_id: {run_id}")
    
    try:
        # Get authentication headers with token
        headers = get_auth_headers()
        
        # Set up the training request
        training_request = {
            "run_id": run_id,
            "model_name": context['dag_run'].conf.get('model_name', None)
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
        
        return result
    
    except Exception as e:
        print(f"Error training model: {str(e)}")
        # We don't want to fail the DAG if training fails
        return None

# Create the DAG
with DAG(
    'mlops_monitor_data_directory',
    default_args=default_args,
    description='Monitor directory for new CSV files and process them',
    schedule_interval=timedelta(hours=1),  # Run every hour
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'monitoring', 'data'],
) as dag:

    t1 = PythonOperator(
        task_id='find_new_csv_files',
        python_callable=find_new_csv_files,
        provide_context=True,
    )

    t2 = PythonOperator(
        task_id='process_csv_files',
        python_callable=process_csv_files,
        provide_context=True,
    )
    
    t3 = PythonOperator(
        task_id='trigger_training_if_needed',
        python_callable=trigger_training_if_needed,
        provide_context=True,
    )

    t1 >> t2 >> t3
