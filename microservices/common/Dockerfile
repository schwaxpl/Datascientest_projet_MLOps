FROM python:3.11-slim

WORKDIR /app

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Installer le module common en mode développement
RUN pip install -e .

CMD ["echo", "This is a base image for other microservices"]
