ARG DOCKERHUB_USERNAME=chameldst
FROM ${DOCKERHUB_USERNAME}/dst-mlops-common:latest

WORKDIR /app

# Copier les fichiers de dépendances spécifiques à l'API de données
COPY microservices/data_api/requirements.txt ./data-requirements.txt

# Installation des dépendances spécifiques
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r data-requirements.txt

# Installation du modèle français pour spaCy et téléchargement des ressources NLTK
RUN python -m spacy download fr_core_news_sm && \
    python -c "import nltk; nltk.download('stopwords')" && \
    python -c "import nltk; nltk.download('punkt')"

# Copier le code source de l'API de données
COPY microservices/data_api/ /app/microservices/data_api/

# Copier le fichier liste_fr.txt dans le répertoire racine pour le rendre accessible
COPY microservices/data_api/liste_fr.txt /app/liste_fr.txt

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
