"""
Module de prédiction utilisant le modèle TF-IDF entraîné.
"""

import pickle
import numpy as np
import pandas as pd
import time
from typing import Union, List, Optional
from src.logger_config import get_logger

# Configuration du logger spécifique au module de prédiction
logger = get_logger('predict')

class PredictionService:
    @classmethod
    def from_artifacts(cls, model, vectorizer):
        """
        Crée une instance du service avec un modèle et un vectorizer déjà chargés.
        
        Args:
            model: Le modèle déjà chargé
            vectorizer: Le vectorizer déjà chargé
            
        Returns:
            PredictionService: Nouvelle instance du service
        """
        logger.info("Création d'une instance PredictionService à partir d'artefacts chargés")
        instance = cls.__new__(cls)
        instance.model = model
        instance.vectorizer = vectorizer
        logger.debug(f"Instance créée avec modèle: {type(model).__name__}, vectorizer: {type(vectorizer).__name__}")
        return instance

    def __init__(self, model_path: str, vectorizer_path: str):
        """
        Initialise le service de prédiction.
        
        Args:
            model_path (str): Chemin vers le fichier du modèle
            vectorizer_path (str): Chemin vers le fichier du vectorizer TF-IDF
        """
        logger.info(f"Initialisation du service de prédiction - model_path: {model_path}, vectorizer_path: {vectorizer_path}")
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
        start_time = time.time()
        self.model = self._load_model()
        self.vectorizer = self._load_vectorizer()
        load_time = time.time() - start_time
        logger.info(f"Service de prédiction initialisé en {load_time:.3f}s")
        
    def _load_model(self):
        """
        Charge le modèle depuis le fichier.
        
        Returns:
            object: Modèle chargé
        """
        logger.info(f"Chargement du modèle depuis {self.model_path}")
        start_time = time.time()
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Modèle chargé en {time.time() - start_time:.3f}s - Type: {type(model).__name__}")
            return model
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}", exc_info=True)
            raise
            
    def _load_vectorizer(self):
        """
        Charge le vectorizer depuis le fichier.
        
        Returns:
            object: Vectorizer TF-IDF chargé
        """
        logger.info(f"Chargement du vectorizer depuis {self.vectorizer_path}")
        start_time = time.time()
        try:
            with open(self.vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            logger.info(f"Vectorizer chargé en {time.time() - start_time:.3f}s - Type: {type(vectorizer).__name__}")
            return vectorizer
        except Exception as e:
            logger.error(f"Erreur lors du chargement du vectorizer: {str(e)}", exc_info=True)
            raise
    
    def _prepare_input(self, text: Union[str, pd.Series, List[str]]) -> np.ndarray:
        """
        Prépare l'entrée pour la prédiction.
        
        Args:
            text (Union[str, pd.Series, List[str]]): Texte à classifier
            
        Returns:
            np.ndarray: Données vectorisées
        """
        logger.debug(f"Préparation des données d'entrée - Type: {type(text).__name__}")
        
        # Convertir le texte en format approprié si nécessaire
        if isinstance(text, str):
            logger.debug(f"Conversion de string en liste - Texte: '{text[:50]}...'")
            text = [text]
        elif isinstance(text, pd.Series):
            logger.debug(f"Conversion de pandas.Series en liste - Taille: {len(text)}")
            text = text.tolist()
            
        # Logging du nombre d'échantillons à traiter
        logger.debug(f"Vectorisation de {len(text)} texte(s)")
        
        # Mesure du temps de vectorisation
        start_time = time.time()
        
        # Vectorisation du texte avec TF-IDF
        try:
            vectors = self.vectorizer.transform(text)
            logger.debug(f"Vectorisation terminée en {time.time() - start_time:.3f}s - Shape: {vectors.shape}")
            return vectors
        except Exception as e:
            logger.error(f"Erreur lors de la vectorisation: {str(e)}", exc_info=True)
            raise
    
    def predict(self, text: Union[str, pd.Series, List[str]]) -> np.ndarray:
        """
        Fait une prédiction pour le texte donné.
        
        Args:
            text (Union[str, pd.Series, List[str]]): Texte à classifier
            
        Returns:
            np.ndarray: Prédictions (0 pour négatif, 1 pour positif)
        """
        logger.info(f"Prédiction de classe pour {type(text).__name__}")
        
        # Mesure du temps de prédiction
        start_time = time.time()
        
        # Préparation des données
        X = self._prepare_input(text)
        
        # Prédiction
        try:
            predictions = self.model.predict(X)
            pred_time = time.time() - start_time
            
            # Calcul de statistiques sur les prédictions
            n_samples = len(predictions) if hasattr(predictions, '__len__') else 1
            n_positive = np.sum(predictions == 1) if hasattr(predictions, '__iter__') else (1 if predictions == 1 else 0)
            
            logger.info(f"Prédiction terminée en {pred_time:.3f}s - {n_samples} échantillon(s), {n_positive} positif(s)")
            return predictions
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}", exc_info=True)
            raise
    
    def predict_proba(self, text: Union[str, pd.Series, List[str]]) -> np.ndarray:
        """
        Retourne les probabilités de prédiction pour chaque classe.
        
        Args:
            text (Union[str, pd.Series, List[str]]): Texte à classifier
            
        Returns:
            np.ndarray: Tableau de probabilités pour chaque classe [p(négatif), p(positif)]
        """
        logger.info(f"Prédiction de probabilités pour {type(text).__name__}")
        
        # Mesure du temps de prédiction
        start_time = time.time()
        
        # Préparation des données
        X = self._prepare_input(text)
        
        try:
            # Prédiction avec le modèle
            pred = self.model.predict(X)
            pred_time = time.time() - start_time
            
            # Log des statistiques de prédiction
            logger.info(f"Prédiction de probabilités terminée en {pred_time:.3f}s - Shape: {pred.shape if hasattr(pred, 'shape') else 'N/A'}")
            return pred
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction de probabilités: {str(e)}", exc_info=True)
            raise
