"""
Module pour équilibrer les datasets déséquilibrés.
Offre des méthodes pour analyser et équilibrer la distribution des avis positifs et négatifs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import uuid
import time
from src.logger_config import get_logger

# Configuration du logger
logger = get_logger('data_balancing')

def analyze_data_distribution(data: pd.DataFrame) -> Dict:
    """
    Analyse la distribution des avis positifs/négatifs dans les données.
    
    Args:
        data (pd.DataFrame): Données à analyser
    
    Returns:
        Dict: Statistiques sur la distribution
    """
    # Vérifier que les colonnes requises sont présentes
    if 'Note' not in data.columns:
        return {'error': 'Column Note not found in data'}
    
    # Distribution des notes
    notes_distribution = data['Note'].value_counts().to_dict()
    
    # Conversion en positif (>3) / négatif (<=3)
    from src.config import POSITIVE_REVIEW_THRESHOLD
    data['sentiment'] = (data['Note'] > POSITIVE_REVIEW_THRESHOLD).astype(int)
    sentiment_counts = data['sentiment'].value_counts()
    
    # Calculer les pourcentages
    total = len(data)
    if 1 in sentiment_counts:
        positive_percent = 100.0 * sentiment_counts[1] / total
    else:
        positive_percent = 0
        
    if 0 in sentiment_counts:
        negative_percent = 100.0 * sentiment_counts[0] / total
    else:
        negative_percent = 0
    
    return {
        'total': total,
        'notes_distribution': notes_distribution,
        'positive': int(sentiment_counts.get(1, 0)),
        'negative': int(sentiment_counts.get(0, 0)),
        'positive_percent': float(positive_percent),
        'negative_percent': float(negative_percent)
    }

def balance_dataset(data: pd.DataFrame, strategy: str = 'hybrid', target_ratio: float = 0.5, random_seed: int = 42) -> pd.DataFrame:
    """
    Équilibre un dataset déséquilibré entre avis positifs et négatifs.
    
    Args:
        data (pd.DataFrame): Données à équilibrer
        strategy (str): Stratégie d'équilibrage ('undersample', 'oversample', 'hybrid')
        target_ratio (float): Ratio cible pour la classe minoritaire (0 à 1)
        random_seed (int): Graine pour la reproductibilité
    
    Returns:
        pd.DataFrame: Données équilibrées
    """
    # Générer un ID unique pour cette opération
    op_id = str(uuid.uuid4())[:8]
    logger.info(f"[{op_id}] Équilibrage du dataset avec stratégie: {strategy}, ratio cible: {target_ratio}")
    
    # Vérifier que les colonnes requises sont présentes
    if 'Note' not in data.columns:
        logger.error(f"[{op_id}] Colonne 'Note' non trouvée dans les données")
        raise ValueError("La colonne 'Note' est nécessaire pour l'équilibrage")
    
    # Classifier les avis en positifs/négatifs
    from src.config import POSITIVE_REVIEW_THRESHOLD
    data = data.copy()  # Éviter les avertissements de modification
    data['sentiment'] = (data['Note'] > POSITIVE_REVIEW_THRESHOLD).astype(int)
    
    # Séparer les avis positifs et négatifs
    positifs = data[data['sentiment'] == 1]
    negatifs = data[data['sentiment'] == 0]
    
    # Mesurer le déséquilibre initial
    nb_pos = len(positifs)
    nb_neg = len(negatifs)
    total = len(data)
    ratio_initial = nb_neg / total if total > 0 else 0
    
    logger.info(f"[{op_id}] Distribution initiale: {nb_pos} positifs, {nb_neg} négatifs (ratio négatifs: {ratio_initial:.4f})")
    
    # Si le dataset est déjà équilibré selon le ratio cible, retourner tel quel
    if abs(ratio_initial - target_ratio) < 0.01:
        logger.info(f"[{op_id}] Le dataset est déjà suffisamment équilibré (ratio: {ratio_initial:.4f}, cible: {target_ratio:.4f})")
        return data
    
    # Appliquer la stratégie d'équilibrage choisie
    if strategy == 'undersample':
        # Sous-échantillonner la classe majoritaire (positifs)
        if target_ratio > 0:
            # Calcul du nombre de positifs à conserver pour atteindre le ratio cible
            # ratio = nb_neg / (nb_neg + nb_pos_new)
            # nb_pos_new = nb_neg * (1 - ratio) / ratio
            nb_pos_new = int(nb_neg * (1 - target_ratio) / target_ratio)
            nb_pos_new = min(nb_pos_new, nb_pos)  # Ne pas dépasser le nombre disponible
            
            # Sous-échantillonner les positifs
            positifs = positifs.sample(nb_pos_new, random_state=random_seed)
            
            logger.info(f"[{op_id}] Sous-échantillonnage: réduit à {nb_pos_new} avis positifs pour équilibrer")
        else:
            logger.warning(f"[{op_id}] Ratio cible de 0 impossible, conservation de tous les avis négatifs uniquement")
            return negatifs
    
    elif strategy == 'oversample':
        # Sur-échantillonner la classe minoritaire (négatifs)
        if target_ratio < 1:
            # Calcul du nombre de négatifs à atteindre pour le ratio cible
            # ratio = nb_neg_new / (nb_neg_new + nb_pos)
            # nb_neg_new = nb_pos * ratio / (1 - ratio)
            nb_neg_new = int(nb_pos * target_ratio / (1 - target_ratio))
            
            # Sur-échantillonner les négatifs
            if nb_neg_new > nb_neg:
                # Calculer combien de fois nous devons répliquer les négatifs
                multiplier = int(nb_neg_new / nb_neg)
                remainder = nb_neg_new % nb_neg
                
                # Répliquer les négatifs
                negatifs_list = [negatifs] * multiplier
                if remainder > 0:
                    negatifs_list.append(negatifs.sample(remainder, random_state=random_seed+1))
                
                negatifs = pd.concat(negatifs_list, ignore_index=True)
                logger.info(f"[{op_id}] Sur-échantillonnage: augmenté à {len(negatifs)} avis négatifs pour équilibrer")
        else:
            logger.warning(f"[{op_id}] Ratio cible de 1 impossible, conservation de tous les avis positifs uniquement")
            return positifs
    
    elif strategy == 'hybrid':
        # Combiner sous-échantillonnage et sur-échantillonnage pour minimiser la perte de données
        # Déterminer le nombre idéal d'avis positifs et négatifs
        total_ideal = min(total, int(nb_neg / target_ratio)) if target_ratio > 0 else total
        nb_neg_ideal = int(total_ideal * target_ratio)
        nb_pos_ideal = total_ideal - nb_neg_ideal
        
        # Ajuster les positifs et négatifs pour atteindre ces nombres
        if nb_pos_ideal < nb_pos:
            # Sous-échantillonner les positifs
            positifs = positifs.sample(nb_pos_ideal, random_state=random_seed)
            logger.info(f"[{op_id}] Hybrid - Sous-échantillonnage positifs: {nb_pos} -> {nb_pos_ideal}")
        
        if nb_neg_ideal > nb_neg:
            # Sur-échantillonner les négatifs
            multiplier = int(nb_neg_ideal / nb_neg)
            remainder = nb_neg_ideal % nb_neg
            
            negatifs_list = [negatifs] * multiplier
            if remainder > 0:
                negatifs_list.append(negatifs.sample(remainder, random_state=random_seed+1))
            
            negatifs = pd.concat(negatifs_list, ignore_index=True)
            logger.info(f"[{op_id}] Hybrid - Sur-échantillonnage négatifs: {nb_neg} -> {len(negatifs)}")
    
    else:
        logger.error(f"[{op_id}] Stratégie d'équilibrage inconnue: {strategy}")
        raise ValueError(f"Stratégie d'équilibrage inconnue: {strategy}")
    
    # Combiner les positifs et négatifs équilibrés
    balanced_data = pd.concat([positifs, negatifs], ignore_index=True)
    
    # Mélanger les données
    balanced_data = balanced_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Mesurer l'équilibre final
    nb_pos_final = sum(balanced_data['sentiment'] == 1)
    nb_neg_final = sum(balanced_data['sentiment'] == 0)
    ratio_final = nb_neg_final / len(balanced_data) if len(balanced_data) > 0 else 0
    
    logger.info(f"[{op_id}] Distribution finale: {nb_pos_final} positifs, {nb_neg_final} négatifs (ratio négatifs: {ratio_final:.4f})")
    
    # Supprimer la colonne sentiment temporaire
    balanced_data = balanced_data.drop(columns=['sentiment'])
    
    return balanced_data
