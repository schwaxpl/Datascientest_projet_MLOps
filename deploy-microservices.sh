#!/bin/bash

echo "Déploiement de l'architecture microservices MLOps..."

# Vérifier que Docker est en cours d'exécution
if ! docker info > /dev/null 2>&1; then
  echo "Erreur: Docker n'est pas en cours d'exécution."
  echo "Veuillez démarrer Docker et réessayer."
  exit 1
fi

# Construire et démarrer les services
echo "Construction et démarrage des conteneurs..."
docker-compose -f docker-compose-microservices.yml up -d --build

# Vérifier que les services sont opérationnels
echo "Attente du démarrage des services..."
sleep 10

echo "Vérification de l'état des services..."
docker-compose -f docker-compose-microservices.yml ps

echo
echo "Architecture microservices déployée!"
echo
echo "Accès aux services:"
echo "- API Gateway: http://localhost:8000"
echo "- Documentation API: http://localhost:8000/docs"
echo "- MLflow: http://localhost:5000"
echo "- MinIO Console: http://localhost:9001"
echo
echo "Utilisateurs par défaut:"
echo "- Admin: admin/adminpassword"
echo "- Utilisateur: user/userpassword"
