#!/bin/bash

echo "Déploiement de l'architecture microservices MLOps..."

# Vérifier que Docker est en cours d'exécution
if ! docker info > /dev/null 2>&1; then
  echo "Erreur: Docker n'est pas en cours d'exécution."
  echo "Veuillez démarrer Docker et réessayer."
  exit 1
fi

# Créer le fichier .env.microservices s'il n'existe pas
if [ ! -f .env.microservices ]; then
  echo "Création du fichier .env.microservices par défaut..."
  cat > .env.microservices << EOF
# Docker Compose environment variables for microservices
# Use this file to control which image versions are pulled from Docker Hub

# Docker Hub username
DOCKERHUB_USERNAME=chameldst

# Microservices image tags
GATEWAY_API_IMAGE_TAG=latest
PREDICTION_API_IMAGE_TAG=latest
TRAINING_API_IMAGE_TAG=latest
DATA_API_IMAGE_TAG=latest
AIRFLOW_IMAGE_TAG=latest
EOF
fi

# Récupérer les dernières images de Docker Hub
echo "Récupération des images Docker depuis Docker Hub..."
docker-compose -f docker-compose-microservices.yml --env-file .env.microservices pull

# Démarrer les services
echo "Démarrage des conteneurs..."
docker-compose -f docker-compose-microservices.yml --env-file .env.microservices up -d

# Vérifier que les services sont opérationnels
echo "Attente du démarrage des services..."
sleep 10

echo "Vérification de l'état des services..."
docker-compose -f docker-compose-microservices.yml --env-file .env.microservices ps

echo
echo "Architecture microservices déployée!"
echo
echo "Accès aux services:"
echo "- API Gateway: http://localhost:8000"
echo "- Documentation API: http://localhost:8000/docs"
echo "- Airflow: http://localhost:8080 (admin/admin)"
echo "- MLflow: http://localhost:5000"
echo "- MinIO Console: http://localhost:9001"
echo
echo "Utilisateurs par défaut:"
echo "- Admin: admin/adminpassword"
echo "- Utilisateur: user/userpassword"
