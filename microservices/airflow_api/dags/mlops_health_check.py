"""
MLOps Health Check DAG

Ce DAG effectue des vérifications périodiques de la santé des différentes API 
de l'architecture MLOps:
- Gateway API
- Data API
- Training API
- Prediction API
- MLflow

Il enregistre les métriques et envoie des alertes en cas de problème.
"""

from datetime import datetime, timedelta
import os
import json
import requests
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
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

# Configuration des URLs
try:
    GATEWAY_API_URL = Variable.get("GATEWAY_API_URL")
except KeyError:
    GATEWAY_API_URL = "http://gateway-api:8000"

# Endpoints de santé pour chaque service
HEALTH_ENDPOINTS = {
    "gateway": f"{GATEWAY_API_URL}/health",
    "data": f"{GATEWAY_API_URL}/data/health",
    "training": f"{GATEWAY_API_URL}/training/health",
    "prediction": f"{GATEWAY_API_URL}/prediction/health",
    "mlflow": "http://mlflow:5000/health"
}

# Temps maximal de réponse acceptable (secondes)
try:
    MAX_RESPONSE_TIME = float(Variable.get("MAX_RESPONSE_TIME"))
except KeyError:
    MAX_RESPONSE_TIME = 5.0

# Récupérer les en-têtes d'authentification
def get_auth_headers():
    """Get authentication headers using username and password from Airflow connections"""
    try:
        # Essayer de récupérer les identifiants depuis la connexion Airflow
        conn = BaseHook.get_connection("gateway_api")
        username = conn.login
        password = conn.password
        
        # Utiliser les variables d'environnement si disponibles
        if not username or not password:
            username = os.environ.get("ADMIN_USERNAME", "admin")
            password = os.environ.get("ADMIN_PASSWORD", "adminpassword")
            logging.info("Using environment variables for API authentication")
        
        # Obtenir le jeton JWT depuis l'API Gateway
        auth_response = requests.post(
            f"{GATEWAY_API_URL}/token",
            data={"username": username, "password": password}
        )
        auth_response.raise_for_status()
        token = auth_response.json().get("access_token")
        
        if not token:
            raise ValueError("No token received from authentication endpoint")
            
        logging.info("Successfully obtained authentication token")
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    except Exception as e:
        logging.error(f"Error getting auth headers: {str(e)}")
        # Fallback avec identifiants par défaut
        try:
            auth_response = requests.post(
                f"{GATEWAY_API_URL}/token",
                data={"username": "admin", "password": "adminpassword"}
            )
            auth_response.raise_for_status()
            token = auth_response.json().get("access_token")
            if not token:
                raise ValueError("No token received from authentication endpoint using default credentials")
            logging.info("Using default credentials for API authentication")
            return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        except Exception as inner_e:
            logging.error(f"Critical authentication error: {str(inner_e)}")
            raise

def check_api_health(**context):
    """
    Vérifier la santé de toutes les API
    """
    headers = get_auth_headers()
    results = {}
    
    for service_name, endpoint in HEALTH_ENDPOINTS.items():
        try:
            start_time = datetime.now()
            
            # Certaines API ne nécessitent pas d'authentification
            current_headers = headers if service_name != "mlflow" else {}
            
            response = requests.get(
                endpoint, 
                headers=current_headers, 
                timeout=MAX_RESPONSE_TIME
            )
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Vérifier si la réponse est valide
            is_healthy = response.status_code == 200
            
            # Si la réponse a un format JSON, extraire les détails
            try:
                details = response.json()
            except:
                details = {"raw_response": response.text[:100]} if response.text else {}
                
            results[service_name] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "status_code": response.status_code,
                "response_time": response_time,
                "details": details
            }
            
            logging.info(f"Service {service_name} is {'healthy' if is_healthy else 'unhealthy'} "
                         f"(code: {response.status_code}, time: {response_time:.2f}s)")
            
            # Si la réponse est trop lente, le marquer comme un problème
            if response_time > MAX_RESPONSE_TIME:
                results[service_name]["status"] = "slow"
                logging.warning(f"Service {service_name} is responding slowly: {response_time:.2f}s")
                
        except requests.exceptions.Timeout:
            logging.error(f"Service {service_name} timed out after {MAX_RESPONSE_TIME} seconds")
            results[service_name] = {
                "status": "timeout",
                "status_code": None,
                "response_time": MAX_RESPONSE_TIME,
                "details": {"error": "Request timed out"}
            }
        except Exception as e:
            logging.error(f"Error checking health for service {service_name}: {str(e)}")
            results[service_name] = {
                "status": "error",
                "status_code": None,
                "response_time": None,
                "details": {"error": str(e)}
            }
    
    # Stocker les résultats pour les prochaines tâches
    context['ti'].xcom_push(key='health_check_results', value=results)
    
    # Déterminer l'état global du système
    unhealthy_services = [service for service, data in results.items() 
                          if data["status"] not in ["healthy"]]
    
    if unhealthy_services:
        logging.warning(f"Unhealthy services detected: {', '.join(unhealthy_services)}")
        return False
    else:
        logging.info("All services are healthy")
        return True

def send_alerts(**context):
    """
    Envoyer des alertes si des services sont en mauvaise santé
    """
    results = context['ti'].xcom_pull(key='health_check_results', task_ids='check_api_health')
    
    # Identifier les services en mauvaise santé
    unhealthy_services = {
        service: data for service, data in results.items() 
        if data["status"] not in ["healthy"]
    }
    
    if unhealthy_services:
        alert_message = "MLOps Health Alert:\n"
        for service, data in unhealthy_services.items():
            alert_message += f"- {service}: {data['status']}"
            if data['status_code']:
                alert_message += f" (code: {data['status_code']})"
            if data['response_time']:
                alert_message += f" (time: {data['response_time']:.2f}s)"
            alert_message += "\n"
        
        # Dans un environnement réel, vous enverriez cette alerte par email/Slack/etc.
        logging.warning(f"ALERT: {alert_message}")
        
        # TODO: Implémenter des connecteurs pour envoyer des alertes réelles
        # (email, Slack, etc.)
        
        return alert_message
    else:
        return "All systems operational"

def log_metrics(**context):
    """
    Enregistrer les métriques de santé pour analyse future
    """
    results = context['ti'].xcom_pull(key='health_check_results', task_ids='check_api_health')
    
    # Déterminer l'état global du système
    unhealthy_services = [service for service, data in results.items() 
                         if data["status"] not in ["healthy"]]
    overall_status = "unhealthy" if unhealthy_services else "healthy"
    
    # Pour l'instant, journalisons les temps de réponse
    for service, data in results.items():
        if data['response_time']:
            logging.info(f"METRIC - {service}_response_time: {data['response_time']:.4f}s")
    
    # Créer un enregistrement structuré
    record = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "overall_status": overall_status,
        "alert_sent": len(unhealthy_services) > 0
    }
    
    # Enregistrer les résultats dans un fichier JSON
    try:
        # Créer le dossier s'il n'existe pas
        monitoring_dir = "/opt/airflow/dags/monitoring"
        os.makedirs(monitoring_dir, exist_ok=True)
        
        # Créer un nom de fichier basé sur la date
        filename = f"{monitoring_dir}/health_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Écrire les données au format JSON
        with open(filename, 'w') as f:
            json.dump(record, f, indent=2)
        
        logging.info(f"Health check results saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving health check results: {str(e)}")
    
    return True

# Créer le DAG
dag = DAG(
    'mlops_health_check',
    default_args=default_args,
    description='Monitor the health of MLOps APIs',
    schedule_interval=timedelta(minutes=15),  # Exécuter toutes les 15 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'monitoring'],
)

check_health_task = PythonOperator(
    task_id='check_api_health',
    python_callable=check_api_health,
    provide_context=True,
    dag=dag,
)

send_alerts_task = PythonOperator(
    task_id='send_alerts',
    python_callable=send_alerts,
    provide_context=True,
    dag=dag,
)

log_metrics_task = PythonOperator(
    task_id='log_metrics',
    python_callable=log_metrics,
    provide_context=True,
    dag=dag,
)

# Définir les dépendances
check_health_task >> [send_alerts_task, log_metrics_task]

if __name__ == "__main__":
    dag.cli()
