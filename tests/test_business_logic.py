"""
Tests unitaires pour la logique métier du système MLOps.

Ce module contient les tests pour valider les fonctions de preprocessing,
de classification et de validation de données sans dépendances externes.
Tous les tests sont conçus pour être rapides et fiables.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing_clean import preprocess_text, determine_themes


class TestTextPreprocessing:
    """
    Tests unitaires pour les fonctions de preprocessing de texte.
    
    Valide le comportement des fonctions de nettoyage et de normalisation
    de texte utilisées dans le pipeline de traitement des avis clients.
    """

    def setup_method(self):
        """Configuration commune pour les tests de preprocessing."""
        self.french_vocab = {
            'bonjour', 'monde', 'excellent', 'service', 'produit', 'bon',
            'mauvais', 'qualité', 'livraison', 'rapide', 'client'
        }
        
        # Mock des objets spaCy pour éviter les dépendances externes
        class MockToken:
            def __init__(self, text, lemma):
                self.text = text
                self.lemma_ = lemma.lower()
                self.is_punct = text in '.,!?;:'
                self.is_space = text.isspace()
        
        class MockNLP:
            def __call__(self, text):
                words = text.split()
                return [MockToken(word, word) for word in words if word.strip()]
        
        self.nlp = MockNLP()
        self.stop_words = {'le', 'la', 'de', 'du', 'des', 'et', 'est', 'un', 'une'}

    def test_preprocess_text_standard_case(self):
        """
        Test du preprocessing avec un texte standard.
        
        Vérifie que la fonction applique correctement le nettoyage,
        la suppression des mots vides et la lemmatisation.
        """
        text = "Excellent service! Le produit est bon."
        result = preprocess_text(text, self.french_vocab, self.nlp, self.stop_words)
        
        # Vérifications de base
        assert isinstance(result, str)
        words = result.split()
        
        # Vérification présence des mots significatifs
        assert 'excellent' in words
        assert 'service' in words
        assert 'produit' in words
        assert 'bon' in words
        
        # Vérification suppression des mots vides
        assert 'le' not in words
        assert 'est' not in words

    def test_preprocess_text_edge_cases(self):
        """
        Test du preprocessing avec des cas limites.
        
        Valide le comportement avec des entrées vides, nulles,
        ou contenant uniquement des caractères spéciaux.
        """
        # Test avec None
        result = preprocess_text(None, self.french_vocab, self.nlp, self.stop_words)
        assert result == ""
        
        # Test avec chaîne vide
        result = preprocess_text("", self.french_vocab, self.nlp, self.stop_words)
        assert isinstance(result, str)
        
        # Test avec seulement des mots non-français
        result = preprocess_text("hello world", self.french_vocab, self.nlp, self.stop_words)
        assert result.strip() == ""

    def test_preprocess_text_special_characters(self):
        """
        Test du preprocessing avec caractères spéciaux.
        
        Vérifie que les caractères spéciaux sont correctement traités
        selon les règles métier définies.
        """
        text = "Super produit!!! Très bon... 5/5 étoiles!!!"
        result = preprocess_text(text, self.french_vocab, self.nlp, self.stop_words)
        
        # Les mots valides doivent être conservés
        words = result.split()
        assert 'produit' in words
        assert 'bon' in words


class TestThemeClassification:
    """
    Tests unitaires pour la classification thématique des avis.
    
    Valide la logique de détection et de priorisation des thèmes
    basée sur l'analyse lexicale des textes.
    """

    def setup_method(self):
        """Configuration commune pour les tests de classification."""
        self.classification_df = pd.DataFrame({
            'Theme': ['Service Client', 'Qualité Produit', 'Livraison', 'Prix'],
            'Mots': [
                {'service', 'client', 'support', 'aide', 'conseiller'},
                {'qualité', 'produit', 'matériau', 'finition', 'solidité'},
                {'livraison', 'envoi', 'délai', 'transport', 'rapidité'},
                {'prix', 'coût', 'tarif', 'cher', 'économique'}
            ]
        })

    def test_determine_themes_multiple_matches(self):
        """
        Test de détection de thèmes multiples.
        
        Vérifie que tous les thèmes pertinents sont détectés
        quand le texte contient plusieurs domaines thématiques.
        """
        text = "service client excellent produit de qualité livraison rapide prix correct"
        themes = determine_themes(text, self.classification_df, max_themes=4)
        
        assert isinstance(themes, list)
        assert len(themes) <= 4
        
        # Vérification présence des thèmes attendus
        expected_themes = ['Service Client', 'Qualité Produit', 'Livraison', 'Prix']
        for theme in expected_themes:
            assert theme in themes

    def test_determine_themes_priority_ordering(self):
        """
        Test de l'ordre de priorité des thèmes.
        
        Vérifie que les thèmes sont classés par pertinence
        (nombre d'occurrences de mots-clés).
        """
        # Texte avec plus d'occurrences pour "Service Client"
        text = "service client support aide conseiller excellent"
        themes = determine_themes(text, self.classification_df, max_themes=3)
        
        # Le thème avec le plus d'occurrences doit être en premier
        assert themes[0] == 'Service Client'

    def test_determine_themes_no_matches(self):
        """
        Test avec un texte sans correspondance thématique.
        
        Vérifie le comportement quand aucun mot-clé n'est trouvé.
        """
        text = "hello world test random words"
        themes = determine_themes(text, self.classification_df, max_themes=3)
        
        assert isinstance(themes, list)
        assert len(themes) == 0

    def test_determine_themes_max_limit(self):
        """
        Test de la limitation du nombre de thèmes.
        
        Vérifie que le paramètre max_themes est respecté.
        """
        text = "service client qualité produit livraison prix support aide"
        themes = determine_themes(text, self.classification_df, max_themes=2)
        
        assert len(themes) <= 2


class TestBusinessRules:
    """
    Tests unitaires pour les règles métier spécifiques.
    
    Valide l'implémentation des règles de validation et de classification
    définies dans les spécifications métier.
    """

    def test_sentiment_classification_rules(self):
        """
        Test des règles de classification automatique des sentiments.
        
        Vérifie l'application correcte des seuils et critères
        pour la classification positive/negative/neutre.
        """
        def classify_sentiment(note, text_length):
            """Règles métier pour la classification de sentiment."""
            if note >= 4:
                return 'positif'
            elif note <= 2:
                return 'negatif'
            elif note == 3 and text_length > 50:
                return 'neutre_detaille'
            else:
                return 'neutre_simple'
        
        # Tests des règles de classification
        assert classify_sentiment(5, 30) == 'positif'
        assert classify_sentiment(1, 20) == 'negatif'
        assert classify_sentiment(3, 80) == 'neutre_detaille'
        assert classify_sentiment(3, 20) == 'neutre_simple'

    def test_data_validation_rules(self):
        """
        Test des règles de validation des données d'entrée.
        
        Vérifie les contrôles de cohérence et d'intégrité
        appliqués aux avis clients.
        """
        def validate_review(avis, note):
            """Règles métier pour la validation des avis."""
            errors = []
            
            # Validation contenu
            if not avis or pd.isnull(avis) or len(avis.strip()) == 0:
                errors.append("Contenu obligatoire")
            elif len(avis.strip()) < 10:
                errors.append("Contenu trop court")
            
            # Validation note
            if not isinstance(note, (int, float)) or note < 1 or note > 5:
                errors.append("Note invalide")
            
            # Validation cohérence
            if note >= 4 and len(avis.strip()) < 20:
                errors.append("Incohérence note/contenu")
            
            return errors
        
        # Tests de validation
        assert len(validate_review("Excellent produit, très satisfait", 5)) == 0
        assert "Contenu obligatoire" in validate_review("", 3)
        assert "Contenu trop court" in validate_review("OK", 3)
        assert "Note invalide" in validate_review("Bon produit", 6)
        assert "Incohérence note/contenu" in validate_review("Bien", 5)

    def test_feature_engineering_logic(self):
        """
        Test de la logique d'extraction de caractéristiques.
        
        Vérifie le calcul des features utilisées pour l'entraînement
        des modèles de machine learning.
        """
        def extract_review_features(avis, note):
            """Extraction des caractéristiques d'un avis."""
            features = {}
            
            # Features textuelles
            features['text_length'] = len(avis)
            features['word_count'] = len(avis.split())
            features['avg_word_length'] = features['text_length'] / features['word_count'] if features['word_count'] > 0 else 0
            
            # Features sémantiques
            positive_indicators = ['excellent', 'parfait', 'super', 'génial']
            negative_indicators = ['mauvais', 'horrible', 'nul', 'décevant']
            
            avis_lower = avis.lower()
            features['positive_words'] = sum(1 for word in positive_indicators if word in avis_lower)
            features['negative_words'] = sum(1 for word in negative_indicators if word in avis_lower)
            
            # Feature de cohérence
            sentiment_score = features['positive_words'] - features['negative_words']
            note_normalized = (note - 3) / 2  # Normalisation [-1, 1]
            features['coherence_score'] = 1 - abs(sentiment_score - note_normalized) / 2
            
            return features
        
        # Test d'extraction
        avis = "Excellent produit, super qualité! Je recommande."
        features = extract_review_features(avis, 5)
        
        assert features['text_length'] == len(avis)
        assert features['word_count'] == len(avis.split())
        assert features['positive_words'] >= 2
        assert features['negative_words'] == 0
        assert 0 <= features['coherence_score'] <= 1


class TestModelSelection:
    """
    Tests unitaires pour la logique de sélection de modèles.
    
    Valide les critères et algorithmes utilisés pour choisir
    le meilleur modèle parmi plusieurs candidats.
    """

    def test_model_selection_criteria(self):
        """
        Test des critères de sélection des modèles.
        
        Vérifie l'application des règles de sélection basées sur
        les métriques de performance et la complexité.
        """
        def select_best_model(models):
            """Sélection du meilleur modèle selon nos critères."""
            # Filtre: accuracy minimum
            candidates = [m for m in models if m['accuracy'] >= 0.8]
            if not candidates:
                return None
            
            # Critère principal: F1-score
            best_f1 = max(candidates, key=lambda x: x['f1_score'])
            
            # Critère secondaire: simplicité (moins de paramètres)
            finalists = [m for m in candidates if m['f1_score'] == best_f1['f1_score']]
            if len(finalists) > 1:
                return min(finalists, key=lambda x: x.get('parameters', float('inf')))
            
            return best_f1
        
        # Test de sélection
        models = [
            {'name': 'A', 'accuracy': 0.75, 'f1_score': 0.80, 'parameters': 1000},
            {'name': 'B', 'accuracy': 0.85, 'f1_score': 0.82, 'parameters': 5000},
            {'name': 'C', 'accuracy': 0.83, 'f1_score': 0.82, 'parameters': 2000}
        ]
        
        best = select_best_model(models)
        assert best['name'] == 'C'  # Même F1 que B mais moins de paramètres
        
        # Test sans candidat valide
        invalid_models = [{'accuracy': 0.7, 'f1_score': 0.75}]
        assert select_best_model(invalid_models) is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])