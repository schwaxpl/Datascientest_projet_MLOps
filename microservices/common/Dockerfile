FROM python:3.11-slim

WORKDIR /app

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source dans le dossier microservices/common
COPY . /app/microservices/common/

# Créer un lien symbolique pour la rétrocompatibilité
RUN ln -sf /app/microservices/common /app/common

# Installer le module common en mode développement
WORKDIR /app/microservices/common
RUN pip install -e .
WORKDIR /app

CMD ["echo", "This is a base image for other microservices"]
