"""
Fonctions utilitaires partagées entre les différents modules et services.
"""

import mlflow
from mlflow.tracking import MlflowClient
import tensorflow as tf
import pickle
import os
import boto3
from typing import Optional, Dict
import pandas as pd
import requests
import json
from functools import wraps
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
import time

from microservices.common.config import (
    MLFLOW_TRACKING_URI,
    SECRET_KEY, 
    ALGORITHM, 
    ACCESS_TOKEN_EXPIRE_MINUTES,
    API_USERS
)

# Fonctions MLflow
def get_mlflow_client():
    """
    Crée et retourne un client MLflow configuré pour fonctionner avec MinIO/S3.
    Configure également les variables d'environnement S3 nécessaires.
    
    Returns:
        MlflowClient: Un client MLflow configuré
    """
    # S'assurer que les variables d'environnement sont définies
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Configuration pour S3/MinIO
    if "MLFLOW_S3_ENDPOINT_URL" in os.environ:
        # La configuration existe déjà, vérifier que boto3 est correctement configuré
        boto3_session = boto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
    
    return MlflowClient()

def load_model_from_registry(model_name: str, version: Optional[str] = None, stage: str = "Production"):
    """
    Charge un modèle depuis MLflow Model Registry.
    
    Args:
        model_name (str): Nom du modèle enregistré
        version (Optional[str]): Version spécifique du modèle. Si None, utilise la dernière version du stage spécifié.
        stage (str): Stage MLflow du modèle à charger (Production, Staging, None). Utilisé uniquement si version est None.
        
    Returns:
        object: Le modèle chargé
    """
    client = get_mlflow_client()
    
    if version is not None:
        # Chargement d'une version spécifique du modèle
        model_uri = f"models:/{model_name}/{version}"
    else:
        # Chargement de la dernière version du stage spécifié
        model_uri = f"models:/{model_name}/{stage}"
    
    # Essayer de charger comme un modèle TensorFlow
    try:
        model = mlflow.tensorflow.load_model(model_uri)
    except Exception as e:
        # Si ça échoue, essayer comme un modèle générique
        model = mlflow.pyfunc.load_model(model_uri)
    
    return model

def get_latest_model_version(client: MlflowClient, model_name: str):
    """
    Obtient la dernière version du modèle.
    
    Args:
        client (MlflowClient): Client MLflow
        model_name (str): Nom du modèle
        
    Returns:
        ModelVersion: La dernière version du modèle
    
    Raises:
        ValueError: Si aucune version n'est trouvée
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"Aucune version trouvée pour le modèle {model_name}")
    
    # Trier par numéro de version (convertir en entier pour un tri correct)
    versions.sort(key=lambda x: int(x.version), reverse=True)
    return versions[0]

def get_vectorizer_from_run(run_id: str):
    """
    Récupère le vectorizer stocké comme artefact MLflow pour un run donné.
    
    Args:
        run_id (str): ID du run MLflow
        
    Returns:
        object: Le vectorizer chargé
    """
    client = get_mlflow_client()
    
    # Construire le chemin vers l'artefact du vectorizer
    artifact_path = f"runs:/{run_id}/vectorizer/tf_idf_vectorizer.pkl"
    
    # Télécharger l'artefact localement dans un répertoire temporaire
    local_path = mlflow.artifacts.download_artifacts(artifact_path)
    
    # Charger le vectorizer depuis le fichier téléchargé
    with open(local_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    return vectorizer

# Fonctions pour la communication inter-services

def call_service(url, method="GET", data=None, headers=None, files=None, timeout=180):
    """
    Appelle un autre microservice avec gestion des erreurs.
    
    Args:
        url (str): URL du service
        method (str): Méthode HTTP (GET, POST, PUT, DELETE)
        data (dict): Données à envoyer
        headers (dict): En-têtes HTTP
        files (dict): Fichiers à envoyer
        timeout (int): Timeout en secondes
        
    Returns:
        dict: Réponse du service
        
    Raises:
        HTTPException: En cas d'erreur
    """
    try:
        if method == "GET":
            response = requests.get(url, params=data, headers=headers, timeout=timeout)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, headers=headers, files=files, timeout=timeout)
            else:
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
        elif method == "PUT":
            response = requests.put(url, json=data, headers=headers, timeout=timeout)
        elif method == "DELETE":
            response = requests.delete(url, json=data, headers=headers, timeout=timeout)
        else:
            raise ValueError(f"Méthode non supportée: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Le service n'est pas disponible: {url}"
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Timeout lors de la connexion au service: {url}"
        )
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ressource non trouvée: {url}"
            )
        else:
            try:
                error_detail = response.json().get("detail", str(e))
            except:
                error_detail = str(e)
                
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Erreur du service: {error_detail}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la communication avec le service: {str(e)}"
        )

# Fonctions pour l'authentification et l'autorisation

# Configuration de la sécurité des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    """Vérifie si un mot de passe en clair correspond à un hash."""
    # Pour simplifier, dans cette démo nous comparons directement les mots de passe
    # En production, utilisez pwd_context.verify
    return plain_password == hashed_password

def get_user(username):
    """Récupère un utilisateur depuis la "base de données"."""
    if username in API_USERS:
        return API_USERS[username]
    return None

def authenticate_user(username, password):
    """Authentifie un utilisateur."""
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user["password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Crée un token JWT."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Vérifie la validité du token et récupère l'utilisateur."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Impossible de valider les identifiants",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user = Depends(get_current_user)):
    """Vérifie si l'utilisateur est actif."""
    return current_user

def check_admin_role(user = Depends(get_current_active_user)):
    """Vérifie si l'utilisateur a le rôle admin."""
    if "admin" not in user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'avez pas les droits nécessaires pour accéder à cette ressource"
        )
    return user

# Décorateur pour mesurer le temps d'exécution
def measure_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"Fonction {func.__name__} exécutée en {execution_time:.3f}s")
        return result
    return wrapper
