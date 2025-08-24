@echo off
SETLOCAL EnableDelayedExpansion

REM Script to build and push Docker images for microservices
REM Usage: build-and-push.bat [username] [tag]

REM Default values
SET DOCKERHUB_USERNAME=%1
SET TAG=%2

IF "%DOCKERHUB_USERNAME%"=="" SET DOCKERHUB_USERNAME=chameldst
IF "%TAG%"=="" SET TAG=latest

echo Building and pushing Docker images with username: %DOCKERHUB_USERNAME% and tag: %TAG%

REM Login to Docker Hub
echo Logging in to Docker Hub...
docker login -u %DOCKERHUB_USERNAME%

REM Build and push common image
echo Building common image...
docker build -t %DOCKERHUB_USERNAME%/dst-mlops-common:%TAG% ./microservices/common/
echo Pushing common image...
docker push %DOCKERHUB_USERNAME%/dst-mlops-common:%TAG%

REM Build and push microservice images
FOR %%S IN (gateway_api prediction_api training_api data_api) DO (
    echo Building %%S image...
    docker build -t %DOCKERHUB_USERNAME%/dst-mlops-%%S:%TAG% --build-arg DOCKERHUB_USERNAME=%DOCKERHUB_USERNAME% ./microservices/%%S/
    
    echo Pushing %%S image...
    docker push %DOCKERHUB_USERNAME%/dst-mlops-%%S:%TAG%
)

echo All images built and pushed successfully!
