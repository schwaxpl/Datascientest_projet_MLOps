# Airflow Microservice for MLOps Project

Ce microservice ajoute la fonctionnalité d'orchestration via Apache Airflow à l'architecture MLOps.

## Fonctionnalités

- **Orchestration des tâches** : Planification et exécution automatisée des workflows de ML
- **DAGs pour l'ingestion de données** : Upload automatisé des fichiers CSV vers l'API de données
- **DAGs pour l'entraînement des modèles** : Entraînement automatique basé sur les données ingérées
- **DAGs pour la surveillance** : Surveillance des répertoires pour détecter de nouveaux fichiers CSV

## DAGs disponibles

1. **mlops_data_ingestion_training** : 
   - Upload d'un fichier CSV vers l'API de données
   - Entraînement d'un modèle à partir des données traitées
   - Validation du modèle entraîné

2. **mlops_monitor_data** :
   - Surveillance horaire d'un répertoire pour de nouveaux fichiers CSV
   - Traitement automatique des nouveaux fichiers
   - Option pour déclencher un entraînement après le traitement

## Utilisation

### Démarrage d'Airflow

Le service Airflow est intégré dans la composition Docker avec les autres microservices. Pour le démarrer :

```bash
docker-compose -f docker-compose-microservices.yml up -d airflow
```

### Authentification API Gateway

Tous les appels API passent par l'API Gateway qui nécessite une authentification par token JWT. Les DAGs gèrent automatiquement :
1. L'obtention d'un token d'authentification auprès de l'API Gateway
2. L'utilisation de ce token dans tous les appels d'API
3. La gestion des erreurs d'authentification

### Déclenchement manuel d'un DAG

Pour déclencher manuellement le DAG d'ingestion et d'entraînement :

1. Accédez à l'interface Airflow (http://localhost:8080)
2. Connectez-vous avec les identifiants (admin/admin)
3. Trouvez le DAG `mlops_data_ingestion_training`
4. Cliquez sur "Trigger DAG" et spécifiez les paramètres :
   ```json
   {
     "csv_file_path": "/opt/airflow/dags/data/avis.csv",
     "model_name": "mon_nouveau_modele",
     "auto_approve": false
   }
   ```

### Utilisation du DAG de surveillance

Le DAG `mlops_monitor_data` s'exécute automatiquement toutes les heures. Pour l'utiliser :

1. Placez les fichiers CSV dans le répertoire `/opt/airflow/dags/data/`
2. Les fichiers seront automatiquement traités lors de la prochaine exécution du DAG
3. Les fichiers traités seront déplacés dans `/opt/airflow/dags/data/processed/`

## Variables Airflow

Configurez ces variables dans l'interface Airflow pour personnaliser le comportement des DAGs :

- `GATEWAY_API_URL` : URL de l'API passerelle (défaut: http://gateway-api:8000)
- `DATA_MONITOR_DIR` : Répertoire à surveiller pour les fichiers CSV (défaut: /opt/airflow/dags/data)
- `DATA_PROCESSED_DIR` : Répertoire pour les fichiers traités (défaut: /opt/airflow/dags/data/processed)
- `AUTO_TRAIN` : Entraînement automatique après ingestion (défaut: False)

## Connexions Airflow

Créez une connexion Airflow nommée `gateway_api` pour stocker les identifiants d'authentification :

- **Conn Id**: gateway_api
- **Conn Type**: HTTP
- **Host**: gateway-api
- **Login**: admin (ou votre nom d'utilisateur)
- **Password**: adminpassword (ou votre mot de passe)
- **Port**: 8000
