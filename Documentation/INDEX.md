# Documentation Technique - MLOps Microservices

Ce répertoire contient la documentation technique complète du projet MLOps d'analyse de sentiments.

## 📋 Guide de Navigation

| Document | Description | Audience |
|----------|-------------|----------|
| **[Architecture Microservices](Architecture_Microservices.md)** | Architecture détaillée du système | Développeurs, Architectes |
| **[API Reference](API_Reference.md)** | Guide complet des endpoints API | Intégrateurs, Développeurs |
| **[Docker Hub Deployment](Docker_Hub_Deployment.md)** | Déploiement avec images pré-construites | DevOps, Production |
| **[Docker Volumes Structure](Docker_Volumes_Structure.md)** | Organisation des volumes persistants | Administrateurs système |

## 🏗️ Architecture du Système

Le projet MLOps implémente une architecture microservices moderne pour l'analyse de sentiments, composée de :

> 📖 **Détails complets** : Voir [Architecture Microservices](Architecture_Microservices.md) pour l'architecture technique détaillée

### 🚪 **Point d'Entrée**
- **API Gateway** (Port 8000) - Authentification JWT et routage

### 🔧 **Services Métier**
- **Prediction API** (Port 8001) - Inférence en temps réel
- **Training API** (Port 8002) - Entraînement et validation de modèles
- **Data API** (Port 8003) - Ingestion et préprocessing des données

### 🗄️ **Infrastructure**
- **MLflow** (Port 5000) - Model Registry et experiment tracking
- **MinIO** (Port 9000/9001) - Stockage S3-compatible
- **Airflow** (Port 8080) - Orchestration des workflows (optionnel)

## 🚀 Démarrage Rapide

```bash
# Déploiement complet
docker-compose -f docker-compose-microservices.yml up -d

# Vérification des services
docker-compose -f docker-compose-microservices.yml ps
```

> 📖 **Options de déploiement** : Consultez [Docker Hub Deployment](Docker_Hub_Deployment.md) pour les images pré-construites

## 🔐 Authentification

Le système utilise JWT (JSON Web Tokens) pour sécuriser tous les endpoints :

```bash
# Obtenir un token
curl -X POST "http://localhost:8000/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=admin&password=adminpassword"
```

> 📖 **API complète** : Consultez l'[API Reference](API_Reference.md#authentification) pour tous les endpoints disponibles

## 📊 Monitoring

- **Logs centralisés** avec rotation automatique dans `/logs`
- **Métriques** collectées via logs structurés  
- **Health checks** disponibles sur tous les services

> 📖 **Configuration** : Voir [Docker Volumes Structure](Docker_Volumes_Structure.md) pour l'organisation des logs

## 🎯 Cas d'Usage

Le système est conçu pour :
- **Analyse de sentiments** sur avis clients TrustPilot
- **Pipeline MLOps complet** (ingestion → entraînement → validation → production)
- **Scalabilité horizontale** via architecture microservices
- **Déploiement containerisé** avec Docker

## 🔗 Navigation Rapide

- **🏗️ Architecture** → [Architecture Microservices](Architecture_Microservices.md)
- **📡 APIs** → [API Reference](API_Reference.md)
- **🐳 Déploiement** → [Docker Hub Deployment](Docker_Hub_Deployment.md)
- **💾 Stockage** → [Docker Volumes Structure](Docker_Volumes_Structure.md)

---

*Documentation du projet MLOps Datascientest - Analyse de Sentiments*
