#!/usr/bin/env python
"""
Script de validation des modèles en ligne de commande.
Permet de valider et promouvoir des modèles sans passer par l'API.
"""

import argparse
import json
import sys
import os
import mlflow
from src.model_validation import validate_model, validate_and_promote_model
from src.logger_config import init_logging
from src.config import VALIDATION_THRESHOLD

# Configuration du logger
loggers = init_logging()
logger = loggers['model_validation']

def main():
    parser = argparse.ArgumentParser(description='Validation et promotion de modèles MLflow')
    subparsers = parser.add_subparsers(dest='command', help='Commande à exécuter')
    
    # Commande 'list' pour lister les modèles disponibles
    list_parser = subparsers.add_parser('list', help='Liste les modèles disponibles')
    
    # Commande 'validate' pour valider un ou plusieurs modèles
    validate_parser = subparsers.add_parser('validate', help='Valide un ou plusieurs modèles')
    validate_parser.add_argument('--model', help='Nom du modèle à valider (optionnel)')
    validate_parser.add_argument('--version', help='Version du modèle à valider (optionnel)')
    validate_parser.add_argument('--threshold', type=float, default=VALIDATION_THRESHOLD, 
                                help=f'Seuil de validation (défaut: {VALIDATION_THRESHOLD})')
    validate_parser.add_argument('--approve', action='store_true', 
                                help='Approuver automatiquement les modèles validés')
    
    # Commande 'promote' pour promouvoir un modèle en production
    promote_parser = subparsers.add_parser('promote', help='Valide et promeut un modèle en production')
    promote_parser.add_argument('model', help='Nom du modèle à promouvoir')
    promote_parser.add_argument('version', help='Version du modèle à promouvoir')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Configuration de MLflow
    from src.config import MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    if args.command == 'list':
        list_models()
    
    elif args.command == 'validate':
        result = validate_model(
            model_name=args.model,
            model_version=args.version,
            approve=args.approve,
            threshold=args.threshold
        )
        
        print(json.dumps(result, indent=2))
        
    elif args.command == 'promote':
        result = validate_and_promote_model(args.model, args.version)
        print(json.dumps(result, indent=2))

def list_models():
    """Liste tous les modèles et versions dans MLflow"""
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    models = client.search_registered_models()
    
    for model in models:
        print(f"\n## Modèle: {model.name}")
        
        versions = client.search_model_versions(f"name='{model.name}'")
        print(f"  Versions: {len(versions)}")
        
        for version in versions:
            status = ""
            if version.current_stage == "Production":
                status = "✅ PRODUCTION"
            elif version.current_stage == "Staging":
                status = "🔶 STAGING"
            else:
                status = "📋 ARCHIVE"
                
            tags = client.get_model_version_tags(model.name, version.version)
            status_tag = tags.get("status", "")
            
            if status_tag:
                status += f" ({status_tag})"
                
            # Formatage de la date
            from datetime import datetime
            created_time = datetime.fromtimestamp(version.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M")
            
            print(f"  - v{version.version} | {status} | Créé le: {created_time} | Run: {version.run_id}")

if __name__ == "__main__":
    main()
