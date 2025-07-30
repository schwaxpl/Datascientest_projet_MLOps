# Projet MLOps Datascientest - Analyse de Sentiments

## Aperçu du Projet
Service d'analyse de sentiments basé sur FastAPI pour les avis TrustPilot utilisant Python 3.10.6.

## Stack Technique
- Python 3.10.6
- FastAPI
- MLflow pour le tracking des expériences et la gestion des modèles
- TensorFlow/Keras pour les modèles de prédiction
- Docker pour la conteneurisation

## Fonctionnalités
- Points d'accès API REST pour l'analyse de sentiments
- Prédiction en temps réel des sentiments des avis TrustPilot
- Ingestion et traitement de données automatisés
- Entraînement de nouveaux modèles
- Validation et promotion de modèles en production
- Architecture MLOps évolutive avec tracking des expériences
- Conteneurisation Docker

## Cycle de vie MLOps

### 1. Ingestion des Données
- Upload de fichiers CSV d'avis clients (entraînement ou validation)
- Prétraitement et nettoyage des données
- Stockage des données traitées avec tags MLflow appropriés:
  - "jdd entrainement" pour les données d'apprentissage
  - "jdd validation" pour les données de validation des modèles

### 2. Entraînement des Modèles
- Utilisation des données ingérées pour l'entraînement
- Possibilité d'utiliser un modèle existant comme base
- Enregistrement des modèles entraînés dans MLflow avec tag "à valider"
- Tracking des métriques et paramètres d'entraînement

### 3. Validation des Modèles
- Évaluation des modèles sur des données de validation spécifiques
- Utilisation de jeux de données dédiés (tagués "jdd validation")
- Métriques de qualité : accuracy, precision, recall, F1-score
- Comparaison avec un seuil de qualité minimal configurable
- Approbation manuelle ou automatique des modèles

### 4. Promotion en Production
- Modèles validés peuvent être promus en production
- Tag MLflow "production" pour le modèle actif
- Transition du modèle vers le stage "Production" dans MLflow

## Utilisation
1. Démarrer le serveur FastAPI
2. Accéder à la documentation interactive sur `http://localhost:8000/docs`
3. Utiliser les endpoints pour prédiction, ingestion, entraînement et validation

## Endpoints API

### Prédiction
- `/predict`: Analyse de sentiment en JSON
- `/predict/form`: Formulaire de test pour l'analyse

### Ingestion de Données
- `/upload`: Upload de fichiers CSV d'avis clients pour l'entraînement
- `/upload/validation`: Upload de fichiers CSV d'avis clients pour la validation

### Entraînement
- `/train`: Entraînement de nouveaux modèles via JSON
- `/train/form`: Formulaire pour l'entraînement de modèles

### Validation et Promotion
- `/validate`: Validation de modèles via JSON
- `/validate/form`: Formulaire pour la validation
- `/promote/{model_name}/{model_version}`: Promotion directe d'un modèle en production

## Documentation API
Accédez à la documentation interactive de l'API sur `http://localhost:8000/docs`

## Gestion des Tags MLflow
- **production**: Modèle actuellement en production
- **à valider**: Modèles entraînés en attente de validation
- **jdd entrainement**: Jeux de données utilisés pour l'entraînement
- **jdd validation**: Jeux de données spécifiques pour la validation des modèles

## Contribution
Veuillez lire CONTRIBUTING.md pour les directives de contribution.

## Licence
Licence MIT
