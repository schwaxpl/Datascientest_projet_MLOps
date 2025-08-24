ARG DOCKERHUB_USERNAME=chameldst
FROM ${DOCKERHUB_USERNAME}/dst-mlops-common:latest

WORKDIR /app

# Copier les fichiers de dépendances spécifiques à l'API gateway
COPY requirements.txt ./gateway-requirements.txt

# Installation des dépendances spécifiques
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r gateway-requirements.txt

# Copier le code source de l'API gateway
COPY . /app/microservices/gateway_api/

# Créer les répertoires nécessaires
RUN mkdir -p /app/logs

# Exposition du port
EXPOSE 8000

# Définir les variables d'environnement
ENV PYTHONPATH=/app

# Commande par défaut
CMD ["python", "/app/microservices/gateway_api/main.py"]
