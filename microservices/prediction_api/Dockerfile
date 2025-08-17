ARG DOCKERHUB_USERNAME=schwaxpl
FROM ${DOCKERHUB_USERNAME}/dst-mlops-common:latest

WORKDIR /app

# Copier les fichiers de dépendances spécifiques à l'API de prédiction
COPY requirements.txt ./prediction-requirements.txt

# Installation des dépendances spécifiques
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r prediction-requirements.txt

# Copier le code source de l'API de prédiction
COPY . /app/prediction_api/

# Créer les volumes nécessaires pour les modèles
RUN mkdir -p /app/models /app/src

# Créer les répertoires nécessaires
RUN mkdir -p /app/logs

# Exposition du port
EXPOSE 8001

# Définir les variables d'environnement
ENV PYTHONPATH=/app

# Commande par défaut
CMD ["python", "microservices/prediction_api/main.py"]
