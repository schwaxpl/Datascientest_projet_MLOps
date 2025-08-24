#!/bin/bash

# Script to build and push Docker images for microservices
# Usage: ./build-and-push.sh [username] [tag]

# Default values
DOCKERHUB_USERNAME=${1:-chameldst}
TAG=${2:-latest}

echo "Building and pushing Docker images with username: $DOCKERHUB_USERNAME and tag: $TAG"

# Login to Docker Hub
echo "Logging in to Docker Hub..."
docker login -u $DOCKERHUB_USERNAME

# Build and push common image
echo "Building common image..."
docker build -t $DOCKERHUB_USERNAME/dst-mlops-common:$TAG ./microservices/common/
echo "Pushing common image..."
docker push $DOCKERHUB_USERNAME/dst-mlops-common:$TAG

# Build and push microservice images
for SERVICE in gateway_api prediction_api training_api data_api
do
    echo "Building $SERVICE image..."
    docker build -t $DOCKERHUB_USERNAME/dst-mlops-$SERVICE:$TAG \
        --build-arg DOCKERHUB_USERNAME=$DOCKERHUB_USERNAME \
        ./microservices/$SERVICE/
    
    echo "Pushing $SERVICE image..."
    docker push $DOCKERHUB_USERNAME/dst-mlops-$SERVICE:$TAG
done

echo "All images built and pushed successfully!"
