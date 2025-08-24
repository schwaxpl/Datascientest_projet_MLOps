ARG DOCKERHUB_USERNAME=chameldst
FROM ${DOCKERHUB_USERNAME}/dst-mlops-common:latest

WORKDIR /app

# Copier les fichiers de dépendances spécifiques à l'API de données
COPY microservices/data_api/requirements.txt ./data-requirements.txt

# Installation des dépendances spécifiques
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r data-requirements.txt

# Copier le code source de l'API de données
COPY microservices/data_api/ /app/microservices/data_api/

# Créer les volumes nécessaires pour les données
RUN mkdir -p /app/data

# Le dossier src a déjà été copié dans l'image common

# Créer les répertoires nécessaires
RUN mkdir -p /app/logs
RUN mkdir -p /app/data/processed

# Exposition du port
EXPOSE 8003

# Définir les variables d'environnement
ENV PYTHONPATH=/app

# Commande par défaut
CMD ["python", "/app/microservices/data_api/main.py"]
