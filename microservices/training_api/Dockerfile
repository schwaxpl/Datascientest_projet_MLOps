ARG DOCKERHUB_USERNAME=chameldst
FROM ${DOCKERHUB_USERNAME}/dst-mlops-common:latest

WORKDIR /app

# Copier les fichiers de dépendances spécifiques à l'API d'entrainement
COPY requirements.txt ./training-requirements.txt

# Installation des dépendances spécifiques
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r training-requirements.txt

# Copier le code source de l'API d'entrainement
COPY . /app/microservices/training_api/

# Créer les volumes nécessaires pour les modèles et les données
RUN mkdir -p /app/models /app/src /app/data

# Créer les répertoires nécessaires
RUN mkdir -p /app/logs

# Exposition du port
EXPOSE 8002

# Définir les variables d'environnement
ENV PYTHONPATH=/app

# Commande par défaut
CMD ["python", "/app/microservices/training_api/main.py"]
