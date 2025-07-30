# Instructions pour le déploiement Docker du projet MLOps

Ce document explique comment déployer l'application d'analyse de sentiments en utilisant Docker.

## Prérequis

- Docker et Docker Compose installés sur votre machine
- Git pour cloner le repository

## Étapes de déploiement

### 1. Construction et démarrage des conteneurs

Pour construire et démarrer l'application avec MLflow sur Windows PowerShell, exécutez :

```powershell
# Utilisez l'une des commandes suivantes selon votre installation Docker :

# Si vous utilisez Docker Desktop sur Windows
docker-compose up -d

# OU avec Docker Compose V2 (plus récent)
docker compose up -d
```

Cela va démarrer deux services :
- `mlflow` : Serveur MLflow sur le port 5000
- `api` : API FastAPI sur le port 8042

### 2. Vérification du déploiement

- Interface MLflow : http://localhost:5000
- Documentation API : http://localhost:8042/docs

### 3. Utilisation

Une fois les conteneurs démarrés, vous pouvez :
- Envoyer des requêtes à l'API pour l'analyse de sentiments
- Gérer les modèles via l'interface MLflow
- Consulter les logs dans le dossier `logs`

### 4. Arrêt des services

Pour arrêter les services sur Windows PowerShell, exécutez :

```powershell
# Utilisez l'une des commandes suivantes selon votre installation Docker :

# Si vous utilisez Docker Desktop sur Windows
docker-compose down

# OU avec Docker Compose V2 (plus récent)
docker compose down
```

## Structure des volumes

Le docker-compose utilise un dossier centralisé `docker_volumes` pour persister les données :
- `docker_volumes/mlflow/data` : Stockage des métadonnées MLflow
- `docker_volumes/mlflow/artifacts` : Stockage des artefacts MLflow
- `docker_volumes/app/logs` : Logs de l'application
- `docker_volumes/app/models` : Modèles entraînés
- `docker_volumes/app/data` : Données d'entrée et traitées

Ce dossier est créé automatiquement lors du premier démarrage des conteneurs.

## Variables d'environnement

Les principales variables d'environnement configurables sont :
- `MLFLOW_TRACKING_URI` : URL du serveur MLflow
- `PORT` : Port sur lequel l'API FastAPI écoute (par défaut : 8042)

Vous pouvez les modifier dans le fichier `docker-compose.yml` si nécessaire.
