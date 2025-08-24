@echo off
echo Deploiement de l'architecture microservices MLOps...

REM Verifier que Docker est en cours d'execution
docker info > nul 2>&1
if %errorlevel% neq 0 (
    echo Erreur: Docker n'est pas en cours d'execution.
    echo Veuillez demarrer Docker Desktop et reessayer.
    exit /b 1
)

REM Créer le fichier .env.microservices s'il n'existe pas
if not exist .env.microservices (
    echo Création du fichier .env.microservices par défaut...
    (
        echo # Docker Compose environment variables for microservices
        echo # Use this file to control which image versions are pulled from Docker Hub
        echo.
        echo # Docker Hub username
        echo DOCKERHUB_USERNAME=chameldst
        echo.
        echo # Microservices image tags
        echo GATEWAY_API_IMAGE_TAG=latest
        echo PREDICTION_API_IMAGE_TAG=latest
        echo TRAINING_API_IMAGE_TAG=latest
        echo DATA_API_IMAGE_TAG=latest
    ) > .env.microservices
)

REM Récupérer les dernières images de Docker Hub
echo Récupération des images Docker depuis Docker Hub...
docker-compose -f docker-compose-microservices.yml --env-file .env.microservices pull

REM Démarrer les services
echo Démarrage des conteneurs...
docker-compose -f docker-compose-microservices.yml --env-file .env.microservices up -d

REM Verifier que les services sont opérationnels
echo Attente du demarrage des services...
timeout /t 10 /nobreak > nul

echo Verification de l'etat des services...
docker-compose -f docker-compose-microservices.yml --env-file .env.microservices ps

echo.
echo Architecture microservices deployee!
echo.
echo Acces aux services:
echo - API Gateway: http://localhost:8000
echo - Documentation API: http://localhost:8000/docs
echo - MLflow: http://localhost:5000
echo - MinIO Console: http://localhost:9001
echo.
echo Utilisateurs par defaut:
echo - Admin: admin/adminpassword
echo - Utilisateur: user/userpassword
