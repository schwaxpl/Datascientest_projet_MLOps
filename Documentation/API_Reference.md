# API Reference - MLOps Microservices

Guide de référence complet des APIs du système MLOps d'analyse de sentiments.

> 📖 **Navigation** : [← Retour à l'index](INDEX.md) | [Architecture système →](Architecture_Microservices.md)

## 🚪 API Gateway (Port 8000)

### Authentification

#### POST /token
Obtenir un token JWT d'authentification.

**Request:**
```bash
curl -X POST "http://localhost:8000/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=user&password=userpassword"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Routage des Requêtes

Tous les endpoints suivants nécessitent un token JWT dans le header :
```
Authorization: Bearer <votre-token>
```

## 📊 Data API (via Gateway /data/)

### Upload de Données

#### POST /data/upload
Upload d'un dataset d'entraînement.

#### POST /data/upload/validation  
Upload d'un dataset de validation.

### Gestion des Datasets

#### GET /data/datasets
Liste de tous les datasets disponibles.

#### GET /data/datasets/{id}
Détails d'un dataset spécifique.

#### GET /data/datasets/{id}/download
Téléchargement d'un dataset.

#### POST /data/datasets/balance
Équilibrage d'un dataset déséquilibré.

## 🤖 Training API (via Gateway /train/)

### Entraînement

#### POST /train/start
Démarrage d'un entraînement de modèle.

#### GET /train/status/{run_id}
Status d'un entraînement en cours.

### Validation et Promotion

#### POST /train/validate/{model_uri}
Validation d'un modèle entraîné.

#### POST /train/promote/{model_uri}
Promotion d'un modèle vers la production.

## 🔮 Prediction API (via Gateway /predict/)

### Prédictions

#### POST /predict/single
Prédiction sur un seul texte.

**Request:**
```json
{
  "text": "Ce produit est fantastique !"
}
```

#### POST /predict/batch
Prédiction sur un batch de textes.

**Request:**
```json
{
  "texts": [
    "Excellent service !",
    "Très déçu de cet achat...",
    "Produit correct dans l'ensemble"
  ]
}
```

### Gestion des Modèles

#### GET /predict/model/current
Informations sur le modèle actuellement en production.

#### POST /predict/model/reload
Rechargement du modèle en cache.

## 🧪 MLflow API (via Gateway /models/)

### Model Registry

#### GET /models/
Liste de tous les modèles enregistrés.

#### GET /models/{model_name}/versions
Versions d'un modèle spécifique.

#### POST /models/{model_name}/versions/{version}/transition
Transition d'un modèle vers un nouveau stage.

## 📈 Monitoring Endpoints

### Health Checks

#### GET /health
Status de santé de l'API Gateway.

#### GET /data/health
Status de santé de l'API Data.

#### GET /train/health  
Status de santé de l'API Training.

#### GET /predict/health
Status de santé de l'API Prediction.

## 🔧 Codes de Réponse

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Requête réussie |
| 201 | Created | Ressource créée |
| 400 | Bad Request | Erreur dans la requête |
| 401 | Unauthorized | Token manquant ou invalide |
| 403 | Forbidden | Permissions insuffisantes |
| 404 | Not Found | Ressource non trouvée |
| 422 | Unprocessable Entity | Erreur de validation |
| 500 | Internal Server Error | Erreur serveur |

## 📝 Exemples d'Usage Complet

### Workflow Complet : Upload → Train → Predict

```bash
# 1. Authentification
TOKEN=$(curl -s -X POST "http://localhost:8000/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=admin&password=adminpassword" | jq -r .access_token)

# 2. Upload dataset
curl -X POST "http://localhost:8000/data/upload" \
     -H "Authorization: Bearer $TOKEN" \
     -F "file=@data/training_data.csv" \
     -F "dataset_name=sentiment_analysis_v1"

# 3. Entraînement
curl -X POST "http://localhost:8000/train/start" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"dataset_id": "latest", "model_name": "sentiment_classifier"}'

# 4. Prédiction
curl -X POST "http://localhost:8000/predict/single" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"text": "Ce produit est excellent !"}'
```

---

*Documentation technique du système MLOps d'analyse de sentiments*

## 🔗 Voir Aussi

- **[Architecture Microservices](Architecture_Microservices.md)** - Détails techniques de l'architecture
- **[Docker Hub Deployment](Docker_Hub_Deployment.md)** - Options de déploiement
- **[Docker Volumes Structure](Docker_Volumes_Structure.md)** - Organisation du stockage
- **[← Retour à l'index](INDEX.md)** - Vue d'ensemble de la documentation
