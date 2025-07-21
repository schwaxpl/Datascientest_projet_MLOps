"""
Configuration centralisée pour le projet MLOps.
Contient toutes les constantes et paramètres utilisés dans les différents modules.
"""

import os

# Constantes MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Noms des modèles et expériences
MODEL_NAME = "dst_trustpilot_1"
INGESTION_EXPERIMENT_NAME = "data_ingestion_api"
TRAINING_EXPERIMENT_NAME = "model_training"

# Chemins des fichiers
MODEL_PATH = "models/tf_idf_mdl.pkl"
VECTORIZER_PATH = "models/tf_idf_vectorizer.pkl"

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
