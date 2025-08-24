ARG DOCKERHUB_USERNAME=chameldst
FROM ${DOCKERHUB_USERNAME}/dst-mlops-common:latest

WORKDIR /app

# Copier les fichiers de dépendances spécifiques à l'API de prédiction
COPY microservices/prediction_api/requirements.txt ./prediction-requirements.txt

# Installation des dépendances spécifiques
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r prediction-requirements.txt

# Copier le code source de l'API de prédiction
COPY microservices/prediction_api/ /app/microservices/prediction_api/

# Créer les volumes nécessaires pour les modèles
RUN mkdir -p /app/models

# Créer les répertoires nécessaires
RUN mkdir -p /app/logs

# Le dossier src a déjà été copié dans l'image common

# Exposition du port
EXPOSE 8001

# Définir les variables d'environnement
ENV PYTHONPATH=/app

# Commande par défaut
CMD ["python", "/app/microservices/prediction_api/main.py"]
