# MLOps Microservices - Docker Hub Images

Ce projet utilise désormais des images Docker pré-construites depuis Docker Hub pour ses microservices.

## Architecture des Images

L'architecture des images est la suivante :

- **Image de base commune** : `chameldst/dst-mlops-common:latest`
  - Contient les dépendances et le code partagés par tous les microservices

- **Images des microservices** :
  - `chameldst/dst-mlops-gateway_api:latest` - API Gateway (authentification et routage)
  - `chameldst/dst-mlops-prediction_api:latest` - API de prédiction
  - `chameldst/dst-mlops-training_api:latest` - API d'entraînement
  - `chameldst/dst-mlops-data_api:latest` - API de gestion des données

## Déploiement

Pour déployer les services en utilisant les images Docker Hub :

```bash
# Windows
.\deploy-microservices.bat

# Linux/MacOS
./deploy-microservices.sh
```

## Configuration des versions d'images

Les versions des images utilisées peuvent être configurées dans le fichier `.env.microservices` :

```
# Docker Hub username
DOCKERHUB_USERNAME=chameldst

# Microservices image tags
GATEWAY_API_IMAGE_TAG=latest
PREDICTION_API_IMAGE_TAG=latest
TRAINING_API_IMAGE_TAG=latest
DATA_API_IMAGE_TAG=latest
```

Pour utiliser une version spécifique, par exemple après un tag git v1.0.0 :

```
GATEWAY_API_IMAGE_TAG=v1.0.0
```

## Construction manuelle des images

Si vous souhaitez construire les images localement plutôt que de les récupérer depuis Docker Hub, vous pouvez utiliser :

```bash
docker-compose -f docker-compose-build.yml build
```

Ou utiliser les scripts fournis pour construire et pousser les images :

```bash
# Sur Windows
.\build-and-push.bat [nom_utilisateur] [tag]

# Sur Linux/macOS
./build-and-push.sh [nom_utilisateur] [tag]
```

## Résoudre les problèmes de chemins dans les conteneurs

Si vous rencontrez des erreurs du type `python: can't open file '/app/microservices/gateway_api/main.py': [Errno 2] No such file or directory`, cela signifie que les chemins dans les conteneurs ne correspondent pas à la structure attendue. Voici comment résoudre ce problème :

1. Assurez-vous que vos Dockerfiles copient les fichiers dans les bons répertoires (par exemple `/app/microservices/gateway_api/`)
2. Vérifiez que les commandes CMD dans les Dockerfiles pointent vers les chemins absolus corrects (par exemple `/app/microservices/gateway_api/main.py`)
3. Reconstruisez les images avec les scripts fournis et redéployez vos conteneurs

## Intégration Continue

Les images sont construites et publiées automatiquement sur Docker Hub via GitHub Actions à chaque push sur la branche `main` ou lors de la création d'un tag `v*`.
