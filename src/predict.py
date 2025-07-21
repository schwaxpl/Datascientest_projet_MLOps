"""
Module de prédiction utilisant le modèle TF-IDF entraîné.
"""

import pickle
import numpy as np
import pandas as pd
from typing import Union, List

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
        instance = cls.__new__(cls)
        instance.model = model
        instance.vectorizer = vectorizer
        return instance

    def __init__(self, model_path: str, vectorizer_path: str):
        """
        Initialise le service de prédiction.
        
        Args:
            model_path (str): Chemin vers le fichier du modèle
            vectorizer_path (str): Chemin vers le fichier du vectorizer TF-IDF
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = self._load_model()
        self.vectorizer = self._load_vectorizer()
        
    def _load_model(self):
        """
        Charge le modèle depuis le fichier.
        
        Returns:
            object: Modèle chargé
        """
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)
            
    def _load_vectorizer(self):
        """
        Charge le vectorizer depuis le fichier.
        
        Returns:
            object: Vectorizer TF-IDF chargé
        """
        with open(self.vectorizer_path, 'rb') as f:
            return pickle.load(f)
    
    def _prepare_input(self, text: Union[str, pd.Series, List[str]]) -> np.ndarray:
        """
        Prépare l'entrée pour la prédiction.
        
        Args:
            text (Union[str, pd.Series, List[str]]): Texte à classifier
            
        Returns:
            np.ndarray: Données vectorisées
        """
        # Convertir le texte en format approprié si nécessaire
        if isinstance(text, str):
            text = [text]
        elif isinstance(text, pd.Series):
            text = text.tolist()
            
        # Vectorisation du texte avec TF-IDF
        return self.vectorizer.transform(text)
    
    def predict(self, text: Union[str, pd.Series, List[str]]) -> np.ndarray:
        """
        Fait une prédiction pour le texte donné.
        
        Args:
            text (Union[str, pd.Series, List[str]]): Texte à classifier
            
        Returns:
            np.ndarray: Prédictions (0 pour négatif, 1 pour positif)
        """
        X = self._prepare_input(text)
        return self.model.predict(X)
    
    def predict_proba(self, text: Union[str, pd.Series, List[str]]) -> np.ndarray:
        """
        Retourne les probabilités de prédiction pour chaque classe.
        
        Args:
            text (Union[str, pd.Series, List[str]]): Texte à classifier
            
        Returns:
            np.ndarray: Tableau de probabilités pour chaque classe [p(négatif), p(positif)]
        """
        X = self._prepare_input(text)
        
        # Prédiction avec le modèle
        pred = self.model.predict(X)
        return pred
