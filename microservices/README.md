# Architecture Microservices MLOps

Ce projet est une refonte du projet MLOps initial en utilisant une architecture microservices.

## Structure de l'Architecture

L'architecture se compose des éléments suivants :

1. **API Passerelle (Gateway)**
   - Point d'entrée unique pour tous les clients
   - Gère l'authentification et l'autorisation
   - Achemine les requêtes vers les services appropriés

2. **API de Prédiction**
   - Responsable uniquement des prédictions
   - Charge les modèles entraînés depuis MLflow
   - Fournit des endpoints pour la prédiction unitaire et par lots

3. **API d'Entraînement**
   - Gère l'entraînement de nouveaux modèles
   - Valide les modèles entraînés
   - Promeut les modèles en production

4. **API de Données**
   - Responsable de l'ingestion et du traitement des données
   - Gère les jeux de données d'entraînement et de validation

5. **Services d'Infrastructure**
   - MLflow pour le tracking des expériences et la gestion des modèles
   - MinIO (S3) pour le stockage des artefacts

## Organisation du Code

```
microservices/
├── common/                   # Code partagé entre les services
│   ├── __init__.py
│   ├── config.py             # Configuration centralisée
│   ├── logger_config.py      # Configuration de logging
│   └── utils.py              # Utilitaires partagés
│
├── gateway_api/              # API Passerelle
│   ├── main.py               # Point d'entrée de l'API passerelle
│   ├── Dockerfile
│   └── requirements.txt
│
├── prediction_api/           # Service de prédiction
│   ├── main.py               # Point d'entrée de l'API de prédiction
│   ├── Dockerfile
│   └── requirements.txt
│
├── training_api/             # Service d'entraînement
│   ├── main.py               # Point d'entrée de l'API d'entraînement
│   ├── Dockerfile
│   └── requirements.txt
│
└── data_api/                 # Service de gestion des données
    ├── main.py               # Point d'entrée de l'API de données
    ├── Dockerfile
    └── requirements.txt
```

## Déploiement avec Docker Compose

Le projet utilise Docker Compose pour orchestrer l'ensemble des services.

### Prérequis

- Docker et Docker Compose installés sur votre machine

### Démarrage des Services

Pour démarrer l'ensemble de l'architecture :

```bash
docker-compose -f docker-compose-microservices.yml up -d
```

### Accès aux Services

- **API Passerelle** : http://localhost:8000
- **Documentation API** : http://localhost:8000/docs
- **Interface MLflow** : http://localhost:5000
- **Interface MinIO** : http://localhost:9001

## Authentification

L'API Passerelle utilise JWT (JSON Web Tokens) pour l'authentification.

Pour obtenir un token d'accès :

1. Envoyez une requête POST à `/token` avec les identifiants dans le corps de la requête :
   ```json
   {
     "username": "user",
     "password": "userpassword"
   }
   ```

2. Incluez le token dans l'en-tête `Authorization` de toutes vos requêtes :
   ```
   Authorization: Bearer <votre-token>
   ```

## Cycle de Vie MLOps

Le cycle de vie MLOps reste identique à celui du projet initial :

1. **Ingestion des Données**
   - Upload de fichiers CSV via l'API de données
   - Prétraitement et nettoyage automatiques
   - Stockage des données traitées avec tags appropriés

2. **Entraînement des Modèles**
   - Utilisation des données ingérées pour l'entraînement via l'API d'entraînement
   - Enregistrement des modèles dans MLflow avec tag "à valider"

3. **Validation des Modèles**
   - Évaluation via l'API d'entraînement sur des données de validation
   - Approbation manuelle ou automatique basée sur des métriques de qualité

4. **Promotion en Production**
   - Modèles validés promus en production via l'API d'entraînement
   - Mise à jour automatique du service de prédiction

## Principaux Endpoints

### API Passerelle

- `/token` : Obtention d'un token JWT
- `/me` : Informations sur l'utilisateur authentifié
- `/health` : État de santé de tous les services
- `/api/docs` : Redirection vers la documentation

### API de Prédiction (via la passerelle)

- `/predict` : Prédiction pour un texte
- `/predict/batch` : Prédiction pour plusieurs textes
- `/models` : Liste des modèles disponibles

### API de Données (via la passerelle)

- `/data/upload` : Upload de données d'entraînement
- `/data/upload/validation` : Upload de données de validation
- `/data/datasets` : Liste des jeux de données
- `/data/datasets/{dataset_id}` : Détails d'un jeu de données

### API d'Entraînement (via la passerelle)

- `/train` : Entraînement d'un nouveau modèle
- `/validate` : Validation d'un modèle
- `/promote/{model_name}/{version}` : Promotion d'un modèle en production
- `/training/models` : Liste des modèles entraînés

## Configuration

Les variables d'environnement principales :

- `SECRET_KEY` : Clé secrète pour la génération des JWT
- `ADMIN_USERNAME`, `ADMIN_PASSWORD` : Identifiants administrateur
- `USER_USERNAME`, `USER_PASSWORD` : Identifiants utilisateur standard
- `MLFLOW_TRACKING_URI` : URL du serveur MLflow
- `MLFLOW_S3_ENDPOINT_URL` : URL du serveur MinIO/S3
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` : Identifiants S3/MinIO
