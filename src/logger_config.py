"""
Configuration centralisée du logging pour l'application MLOps.
Fournit un système de logging professionnel avec différents niveaux et formatage.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Création du répertoire pour les logs s'il n'existe pas
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Nom des fichiers de log avec timestamp
current_date = datetime.now().strftime("%Y%m%d")
log_file = os.path.join(log_dir, f'mlops_{current_date}.log')
error_log_file = os.path.join(log_dir, f'mlops_errors_{current_date}.log')

# Format du logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

# Configuration globale
def setup_logging(level=logging.INFO):
    """
    Configure le système de logging global.
    
    Args:
        level: Niveau de logging (default: INFO)
    """
    # Handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Handler pour le fichier de log principal (rotation à 10MB, max 10 fichiers)
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=10)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Handler séparé pour les erreurs
    error_file_handler = RotatingFileHandler(error_log_file, maxBytes=10*1024*1024, backupCount=10)
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)
    
    # Configuration du logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Suppression des handlers existants pour éviter les doublons
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Ajout des handlers au logger racine
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_file_handler)
    
    return root_logger

# Configuration des loggers spécifiques
def get_logger(name, level=None):
    """
    Récupère un logger configuré pour un module spécifique.
    
    Args:
        name: Nom du logger (généralement __name__)
        level: Niveau de logging spécifique (optionnel)
        
    Returns:
        logging.Logger: Logger configuré
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        logger.setLevel(level)
        
    return logger

# Configuration des handlers pour les requêtes HTTP
class RequestFormatter(logging.Formatter):
    """Formatter personnalisé pour les requêtes HTTP"""
    
    def format(self, record):
        record.url = getattr(record, 'url', 'N/A')
        record.method = getattr(record, 'method', 'N/A')
        record.status_code = getattr(record, 'status_code', 'N/A')
        record.response_time = getattr(record, 'response_time', 0)  # Utilisation de 0 comme valeur par défaut
        record.client_ip = getattr(record, 'client_ip', 'N/A')
        
        return super().format(record)

def setup_api_logging():
    """
    Configure le logging spécifique pour l'API.
    Ajoute des handlers pour les requêtes HTTP.
    
    Returns:
        logging.Logger: Logger configuré pour l'API
    """
    # Création du logger API
    api_logger = logging.getLogger('api')
    
    # Format standard pour les messages généraux de l'API
    standard_format = '%(asctime)s - API - %(levelname)s - %(message)s'
    standard_formatter = logging.Formatter(standard_format)
    
    # Fichier pour les logs généraux de l'API
    general_log_file = os.path.join(log_dir, f'api_general_{current_date}.log')
    general_handler = RotatingFileHandler(general_log_file, maxBytes=10*1024*1024, backupCount=10)
    general_handler.setFormatter(standard_formatter)
    
    # Format spécial pour les requêtes HTTP
    api_format = '%(asctime)s - API - %(levelname)s - [%(method)s] %(url)s - %(status_code)s - %(response_time).3fs - %(client_ip)s - %(message)s'
    api_formatter = RequestFormatter(api_format)
    
    # Fichier spécifique pour les logs d'accès HTTP
    api_access_log_file = os.path.join(log_dir, f'api_access_{current_date}.log')
    api_access_handler = RotatingFileHandler(api_access_log_file, maxBytes=10*1024*1024, backupCount=10)
    api_access_handler.setFormatter(api_formatter)
    
    # Le handler d'accès n'est utilisé que pour les requêtes HTTP, nous ajoutons un filtre
    class RequestFilter(logging.Filter):
        def filter(self, record):
            return hasattr(record, 'method') and hasattr(record, 'url')
    
    # Application du filtre
    api_access_handler.addFilter(RequestFilter())
    
    # Ajout des handlers au logger API
    api_logger.addHandler(general_handler)
    api_logger.addHandler(api_access_handler)
    
    return api_logger

# Fonction d'initialisation principale à appeler au démarrage de l'application
def init_logging(api=False, level=logging.INFO):
    """
    Initialise le système de logging complet.
    
    Args:
        api (bool): Si True, configure aussi le logging spécifique à l'API
        level: Niveau de logging global
        
    Returns:
        dict: Dictionnaire contenant les loggers principaux
    """
    # Configuration du logger principal
    root_logger = setup_logging(level)
    
    # Loggers pour les différents modules
    loggers = {
        'root': root_logger,
        'data_ingestion': get_logger('data_ingestion', level),
        'predict': get_logger('predict', level),
        'train': get_logger('train', level),
    }
    
    # Logger spécifique pour l'API si demandé
    if api:
        loggers['api'] = setup_api_logging()
        
    return loggers

# Exemple d'utilisation dans les autres fichiers:
# from src.logger_config import get_logger
# logger = get_logger(__name__)

# Pour initialiser le système complet au démarrage de l'application:
# from src.logger_config import init_logging
# loggers = init_logging(api=True)
