# GitHub Actions pour le projet MLOps

Ce document explique comment configurer et utiliser le workflow GitHub Actions inclus dans ce dépôt.

## Présentation du workflow

Le workflow `docker-push.yml` est configuré pour automatiquement construire une image Docker et la pousser vers Docker Hub lorsque :
- Un push est effectué sur la branche `main`
- Un nouveau tag commençant par `v` est créé (ex: `v1.0.0`)

## Prérequis

Pour utiliser ce workflow, vous devez configurer deux secrets dans votre dépôt GitHub :

1. `DOCKER_HUB_USERNAME` : Votre nom d'utilisateur Docker Hub
2. `DOCKER_HUB_ACCESS_TOKEN` : Un token d'accès personnel Docker Hub (pas votre mot de passe)

### Création d'un token d'accès Docker Hub

1. Connectez-vous à votre compte [Docker Hub](https://hub.docker.com/)
2. Cliquez sur votre avatar en haut à droite, puis sur "Account Settings"
3. Dans le menu latéral, sélectionnez "Security"
4. Cliquez sur "New Access Token"
5. Donnez un nom à votre token (ex: "GitHub CI")
6. Sélectionnez les permissions appropriées (au minimum "Read & Write")
7. Cliquez sur "Generate"
8. **Important** : Copiez immédiatement le token généré, car il ne sera plus affiché par la suite

### Configuration des secrets dans GitHub

1. Dans votre dépôt GitHub, allez dans "Settings" > "Secrets and variables" > "Actions"
2. Cliquez sur "New repository secret"
3. Créez un secret nommé `DOCKER_HUB_USERNAME` avec votre nom d'utilisateur Docker Hub
4. Créez un second secret nommé `DOCKER_HUB_ACCESS_TOKEN` avec le token généré précédemment

## Fonctionnement du workflow

Le workflow exécute les actions suivantes :

1. Checkout du code source
2. Connexion à Docker Hub avec les identifiants fournis
3. Configuration de Docker Buildx pour une construction multi-plateforme
4. Extraction des métadonnées (tags, labels) pour l'image Docker
5. Construction et push de l'image Docker vers Docker Hub

## Nommage et tags des images

Le workflow génère automatiquement plusieurs tags pour chaque image :

- Pour un push sur `main` : `username/mlops-sentiment-analysis:main`
- Pour un tag `v1.0.0` : 
  - `username/mlops-sentiment-analysis:v1.0.0` 
  - `username/mlops-sentiment-analysis:1.0.0`
- Pour chaque build : un tag basé sur le hash court du commit

## Utilisation manuelle du workflow

Si vous souhaitez déclencher manuellement le workflow :

1. Dans votre dépôt GitHub, allez dans l'onglet "Actions"
2. Sélectionnez "Docker Build and Push" dans la liste des workflows
3. Cliquez sur "Run workflow" à droite
4. Sélectionnez la branche sur laquelle exécuter le workflow
5. Cliquez sur "Run workflow"

## Personnalisation

Vous pouvez personnaliser ce workflow en modifiant le fichier `.github/workflows/docker-push.yml`.

Les modifications courantes incluent :
- Changer le nom de l'image Docker
- Modifier les règles de déclenchement (branches, tags)
- Ajouter des étapes de test avant le build
- Configurer des plateformes supplémentaires pour le build multi-architecture
