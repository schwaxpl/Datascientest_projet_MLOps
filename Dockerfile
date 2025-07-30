FROM python:3.10.6-slim
# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .
COPY setup.py .

# Installation des dépendances système et Python
RUN apt-get update && \
    apt-get install -y git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir boto3 s3fs


# Copier le code source
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY models/ /app/models/

# Commande pour lancer l'API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8042"]
