"""
Configuration centralisée pour le projet MLOps en microservices.
Contient toutes les constantes et paramètres utilisés dans les différents modules.
"""

import os

# Constantes MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", "http://mlflow:5000")

# URLs des services
PREDICTION_API_URL = os.getenv("PREDICTION_API_URL", "http://prediction-api:8001")
TRAINING_API_URL = os.getenv("TRAINING_API_URL", "http://training-api:8002")
DATA_API_URL = os.getenv("DATA_API_URL", "http://data-api:8003")
GATEWAY_API_URL = os.getenv("GATEWAY_API_URL", "http://gateway-api:8000")

# Noms des modèles et expériences
MODEL_NAME = "dst_trustpilot"
INGESTION_EXPERIMENT_NAME = "data_ingestion_api"
TRAINING_EXPERIMENT_NAME = "model_training"

# Chemins des fichiers
MODEL_PATH = "/app/models/tf_idf_mdl.pkl"
# Chemin par défaut pour le vectorizer
VECTORIZER_DEFAULT_PATH = "/app/models/tf_idf_vectorizer.pkl"

# Chemin alternatif pour le vectorizer dans le volume persistant
VECTORIZER_PERSISTENT_PATH = "/app/models_persistent/tf_idf_vectorizer.pkl"

# Utilisation du premier chemin qui existe
import os
VECTORIZER_PATH = VECTORIZER_DEFAULT_PATH if os.path.exists(VECTORIZER_DEFAULT_PATH) else VECTORIZER_PERSISTENT_PATH

# Paramètres d'entraînement
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42
TRAINING_EPOCHS = 5
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Architecture du modèle
MODEL_ARCHITECTURE = {
    "dense_layers": [256, 128, 64],
    "dropout_rates": [0.3, 0.3, 0.2],
    "activation": "relu",
    "output_activation": "softmax"
}

# Colonnes requises dans les données
REQUIRED_COLUMNS = ["Avis", "Note"]

# Seuils de classification
POSITIVE_REVIEW_THRESHOLD = 3  # Note > 3 est considérée comme positive

# Seuils de validation et promotion de modèles
VALIDATION_THRESHOLD = 0.75  # Accuracy minimale pour valider un modèle

# Clé secrète pour la sécurité (JWT)
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-jwt-tokens")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Credentials pour l'authentification de base (à remplacer par une base de données)
API_USERS = {
    "admin": {
        "username": os.getenv("ADMIN_USERNAME", "admin"),
        "password": os.getenv("ADMIN_PASSWORD", "password"),
        "roles": ["admin"]
    },
    "user": {
        "username": os.getenv("USER_USERNAME", "user"),
        "password": os.getenv("USER_PASSWORD", "password"),
        "roles": ["user"]
    }
}
