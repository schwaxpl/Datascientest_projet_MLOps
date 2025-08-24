ARG DOCKERHUB_USERNAME=schwaxpl
FROM ${DOCKERHUB_USERNAME}/dst-mlops-common:latest

WORKDIR /app

# Copier les fichiers de dépendances spécifiques à l'API de données
COPY requirements.txt ./data-requirements.txt

# Installation des dépendances spécifiques
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r data-requirements.txt

# Copier le code source de l'API de données
COPY . /app/microservices/data_api/

# Créer les volumes nécessaires pour les données
RUN mkdir -p /app/data /app/src

# Créer les répertoires nécessaires
RUN mkdir -p /app/logs
RUN mkdir -p /app/data/processed

# Exposition du port
EXPOSE 8003

# Définir les variables d'environnement
ENV PYTHONPATH=/app

# Commande par défaut
CMD ["python", "/app/microservices/data_api/main.py"]
