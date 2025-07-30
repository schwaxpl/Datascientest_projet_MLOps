# Docker Volumes

Ce dossier est utilisé par Docker pour stocker les données persistantes des conteneurs.
Il est automatiquement créé lors du premier démarrage des conteneurs.

## Structure des répertoires

```
docker_volumes/
├── mlflow/
│   ├── data/       # Métadonnées MLflow
│   └── artifacts/  # Artefacts des modèles MLflow
└── app/
    ├── logs/       # Logs de l'application
    ├── models/     # Modèles entraînés
    └── data/       # Données d'entrée et traitées
```

## Note importante

Ne supprimez pas ce dossier si vous souhaitez conserver les données de votre application,
comme les modèles entraînés et les logs.

Si vous souhaitez effectuer une réinitialisation complète, vous pouvez supprimer
ce dossier et redémarrer les conteneurs avec `docker-compose up -d`.
