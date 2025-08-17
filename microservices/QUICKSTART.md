# Guide de Démarrage Rapide - Microservices MLOps

Ce guide vous permettra de démarrer rapidement avec l'architecture microservices du projet MLOps.

## Prérequis

- Docker et Docker Compose installés
- Git installé pour cloner le dépôt

## Démarrage

1. **Cloner le dépôt** (si ce n'est pas déjà fait)
   ```bash
   git clone <url-du-depot>
   cd MLOps
   ```

2. **Lancer l'architecture microservices**
   ```bash
   docker-compose -f docker-compose-microservices.yml up -d
   ```

3. **Vérifier que tous les services sont en cours d'exécution**
   ```bash
   docker-compose -f docker-compose-microservices.yml ps
   ```

## Utilisation de l'API

### 1. Obtenir un Token d'Authentification

```bash
curl -X POST "http://localhost:8000/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=user&password=userpassword"
```

Réponse :
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 2. Télécharger un Jeu de Données

```bash
curl -X POST "http://localhost:8000/data/upload" \
     -H "Authorization: Bearer <votre-token>" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/chemin/vers/votre/fichier.csv"
```

### 3. Entraîner un Modèle

```bash
curl -X POST "http://localhost:8000/train" \
     -H "Authorization: Bearer <votre-token>" \
     -H "Content-Type: application/json" \
     -d '{
           "dataset_id": "processed_data_20250721_001436.csv",
           "model_name": "sentiment_model",
           "params": {
             "max_features": 5000,
             "ngram_range": [1, 2]
           }
         }'
```

### 4. Valider un Modèle

```bash
curl -X POST "http://localhost:8000/validate" \
     -H "Authorization: Bearer <votre-token>" \
     -H "Content-Type: application/json" \
     -d '{
           "model_name": "sentiment_model",
           "version": 1,
           "validation_dataset": "validation.csv"
         }'
```

### 5. Promouvoir un Modèle en Production

```bash
curl -X POST "http://localhost:8000/promote/sentiment_model/1" \
     -H "Authorization: Bearer <votre-token>"
```

### 6. Faire une Prédiction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Authorization: Bearer <votre-token>" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Ce jeu est vraiment fantastique, j'adore y jouer!",
           "model_name": "sentiment_model"
         }'
```

## Accès aux Interfaces Web

- **Documentation de l'API** : http://localhost:8000/docs
- **Interface MLflow** : http://localhost:5000
- **Console MinIO** : http://localhost:9001 (credentials: minio_access_key / minio_secret_key)

## Commandes Docker Utiles

- **Arrêter tous les services**
  ```bash
  docker-compose -f docker-compose-microservices.yml down
  ```

- **Voir les logs d'un service spécifique**
  ```bash
  docker-compose -f docker-compose-microservices.yml logs -f gateway_api
  ```

- **Redémarrer un service spécifique**
  ```bash
  docker-compose -f docker-compose-microservices.yml restart prediction_api
  ```

## Résolution des Problèmes Courants

### 1. Services ne démarrant pas

Vérifiez les logs du service concerné :
```bash
docker-compose -f docker-compose-microservices.yml logs -f <nom_service>
```

### 2. Erreur d'authentification

Vérifiez que vous utilisez le bon token et qu'il n'a pas expiré. Les tokens ont une durée de validité de 30 minutes.

### 3. Problèmes de réseau entre services

Vérifiez que tous les services sont dans le même réseau Docker :
```bash
docker network inspect microservices_network
```
