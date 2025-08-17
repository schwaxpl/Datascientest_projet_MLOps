@echo off
echo Deploiement de l'architecture microservices MLOps...

REM Verifier que Docker est en cours d'execution
docker info > nul 2>&1
if %errorlevel% neq 0 (
    echo Erreur: Docker n'est pas en cours d'execution.
    echo Veuillez demarrer Docker Desktop et reessayer.
    exit /b 1
)

REM Construire et demarrer les services
echo Construction et demarrage des conteneurs...
docker-compose -f docker-compose-microservices.yml up -d --build

REM Verifier que les services sont opérationnels
echo Attente du demarrage des services...
timeout /t 10 /nobreak > nul

echo Verification de l'etat des services...
docker-compose -f docker-compose-microservices.yml ps

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
