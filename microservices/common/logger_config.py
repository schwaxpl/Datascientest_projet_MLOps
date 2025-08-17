"""
Configuration centralisée du logging pour l'application MLOps en microservices.
Fournit un système de logging professionnel avec différents niveaux et formatage.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def init_logging(service_name, api=False):
    """
    Initialise le système de logging pour un microservice spécifique.
    
    Args:
        service_name (str): Nom du service (prediction, training, data, gateway)
        api (bool): Si True, configure les loggers spécifiques pour API
        
    Returns:
        dict: Dictionnaire des loggers configurés
    """
    # Création du répertoire pour les logs s'il n'existe pas
    log_dir = os.path.join('/app', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Nom des fichiers de log avec timestamp
    current_date = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f'{service_name}_{current_date}.log')
    error_log_file = os.path.join(log_dir, f'{service_name}_errors_{current_date}.log')

    # Format du logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # Configuration du logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Handler pour le fichier de log général
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)

    # Handler pour le fichier d'erreurs
    error_file_handler = RotatingFileHandler(error_log_file, maxBytes=10*1024*1024, backupCount=5)
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    # Ajout des handlers au logger racine
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_file_handler)

    # Création d'un logger spécifique pour le service
    service_logger = logging.getLogger(service_name)
    
    # Configuration des loggers pour API si nécessaire
    loggers = {service_name: service_logger}
    
    if api:
        # Configuration pour les logs d'accès API
        api_logger = logging.getLogger(f'{service_name}_api')
        
        # Fichier spécifique pour les logs d'accès HTTP
        api_access_log_file = os.path.join(log_dir, f'{service_name}_api_access_{current_date}.log')
        api_general_log_file = os.path.join(log_dir, f'{service_name}_api_general_{current_date}.log')
        
        # Handler pour les logs généraux d'API
        general_handler = RotatingFileHandler(
            api_general_log_file, 
            maxBytes=10*1024*1024, 
            backupCount=5
        )
        general_handler.setFormatter(formatter)
        
        # Handler pour les logs d'accès HTTP avec format spécifique
        api_access_handler = RotatingFileHandler(
            api_access_log_file, 
            maxBytes=10*1024*1024, 
            backupCount=5
        )
        
        api_format = '%(asctime)s - API - %(levelname)s - [%(method)s] %(url)s - %(status_code)s - %(response_time).3fs - %(client_ip)s - %(message)s'
        api_formatter = RequestFormatter(api_format)
        api_access_handler.setFormatter(api_formatter)
        
        # Filtre pour les logs d'accès HTTP
        class RequestFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, 'method') and hasattr(record, 'url')
        
        api_access_handler.addFilter(RequestFilter())
        
        # Ajout des handlers au logger API
        api_logger.addHandler(general_handler)
        api_logger.addHandler(api_access_handler)
        
        loggers['api'] = api_logger
    
    return loggers

class RequestFormatter(logging.Formatter):
    """
    Formateur spécial pour les logs de requêtes HTTP.
    Prend en charge les attributs supplémentaires comme method, url, status_code, etc.
    """
    def format(self, record):
        # Valeurs par défaut pour les champs spécifiques aux requêtes
        if not hasattr(record, 'method'):
            record.method = 'UNKNOWN'
        if not hasattr(record, 'url'):
            record.url = 'UNKNOWN'
        if not hasattr(record, 'status_code'):
            record.status_code = '???'
        if not hasattr(record, 'response_time'):
            record.response_time = 0.0
        if not hasattr(record, 'client_ip'):
            record.client_ip = 'UNKNOWN'
        
        # Utiliser le formateur parent pour le reste
        return super().format(record)

def get_logger(name):
    """
    Récupère un logger par son nom.
    
    Args:
        name (str): Nom du logger
        
    Returns:
        Logger: Logger configuré
    """
    return logging.getLogger(name)
