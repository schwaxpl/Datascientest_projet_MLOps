"""
Module de pipeline d'ingestion de données.
Responsable du chargement et du prétraitement des données d'avis clients.
Utilise MLflow pour le tracking des métriques et paramètres.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dic    def __init__(self, data_path: str, experiment_name: str,
                 is_validation_set: bool = False, original_filename: str = None):
        """
        Initialise le pipeline d'ingestion de données.
        
        Args:
            data_path (str): Chemin vers le fichier de données
            experiment_name (str): Nom de l'expérience MLflow
            is_validation_set (bool): Si True, les données seront taguées comme "jdd validation"
                                     sinon comme "jdd entrainement"
            original_filename (str): Nom du fichier original uploadé par l'utilisateur
        """
        # Génération d'un ID unique pour ce pipeline d'ingestion
        self.pipeline_id = str(uuid.uuid4())[:8]
        logger.info(f"[{self.pipeline_id}] Initialisation du pipeline d'ingestion - Fichier: {data_path}")
        
        self.data_path = data_path
        self.required_columns = REQUIRED_COLUMNS
        self.original_filename = original_filename or os.path.basename(data_path) List, Set
import mlflow
import time
import uuid
from pathlib import Path
import os
import csv

# Vérifier si les modules avancés sont disponibles
ADVANCED_PREPROCESSING_AVAILABLE = False
try:
    import spacy
    from nltk.corpus import stopwords
    ADVANCED_PREPROCESSING_AVAILABLE = True
except ImportError:
    pass
import re
import matplotlib.pyplot as plt
from datetime import datetime
from src.logger_config import get_logger

# Importation des modules nécessaires pour le prétraitement avancé des textes
try:
    import spacy
    from nltk.corpus import stopwords
    ADVANCED_PREPROCESSING_AVAILABLE = True
except ImportError:
    ADVANCED_PREPROCESSING_AVAILABLE = False

# Configuration du logger spécifique au module d'ingestion de données
logger = get_logger('data_ingestion')

from src.config import (
    REQUIRED_COLUMNS,
    INGESTION_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    AUTO_CLEAN_CSV
)

# Fonctions de prétraitement avancé des textes
def preprocess_text(text, french_vocab=None, nlp=None, stop_words=None):
    """
    Prétraite un texte en supprimant les caractères spéciaux, les mots non français,
    les stop words et en appliquant la lemmatisation.
    
    Args:
        text: Texte à prétraiter
        french_vocab: Ensemble des mots français valides
        nlp: Modèle spaCy chargé
        stop_words: Ensemble des mots vides à supprimer
    
    Returns:
        str: Texte prétraité
    """
    if not ADVANCED_PREPROCESSING_AVAILABLE:
        # Version simplifiée si les dépendances ne sont pas disponibles
        if pd.isnull(text):
            return ""
        # Supprimer les caractères spéciaux et mettre en minuscule
        text = re.sub(r'[^\w\s]', ' ', str(text).lower())
        return text.strip()
    
    if pd.isnull(text):
        return ""
        
    # Supprimer les caractères spéciaux et mettre en minuscule
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    
    if french_vocab:
        # Supprimer les mots qui ne sont pas dans le dictionnaire français
        text = " ".join([word for word in text.split() if word in french_vocab])
    
    if nlp and stop_words:
        # Tokenizer, supprimer les stop words et appliquer la lemmatisation avec spaCy
        doc = nlp(text)
        words = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct and not token.is_space]
        return " ".join(words)
    
    return text.strip()

def determine_themes(text, classification_df, max_themes=5):
    """
    Détermine les thèmes abordés dans un texte en fonction d'une classification de mots.
    
    Args:
        text: Texte à analyser
        classification_df: DataFrame contenant la classification des mots par thème
        max_themes: Nombre maximum de thèmes à retourner
    
    Returns:
        List[str]: Liste des thèmes identifiés
    """
    if not ADVANCED_PREPROCESSING_AVAILABLE or pd.isnull(text):
        return []
        
    theme_counts = {}
    for _, row in classification_df.iterrows():
        theme = row['Theme']
        mots = set(row['Mots'])
        count = sum(1 for word in text.split() if word in mots)
        if count > 0:
            theme_counts[theme] = count
            
    # Trier les thèmes par nombre de mots correspondants et sélectionner les top N
    sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    return [theme for theme, _ in sorted_themes[:max_themes]]

def load_french_vocabulary(vocab_path="liste_fr.txt"):
    """
    Charge le vocabulaire français depuis un fichier texte.
    
    Args:
        vocab_path: Chemin vers le fichier de vocabulaire
    
    Returns:
        Set[str]: Ensemble des mots français
    """
    try:
        with open(vocab_path, 'r', encoding='utf-8') as file:
            french_vocab = set(word.strip() for word in file.readlines())
        return french_vocab
    except Exception as e:
        logger.warning(f"Impossible de charger le vocabulaire français depuis {vocab_path}: {str(e)}")
        return set()
        
def load_classification_df(classification_path="Classification_mots.csv"):
    """
    Charge le DataFrame de classification des mots par thème.
    
    Args:
        classification_path: Chemin vers le fichier CSV de classification
    
    Returns:
        pd.DataFrame: DataFrame avec les thèmes et mots associés
    """
    try:
        classification_df = pd.read_csv(classification_path)
        classification_df['Mots'] = classification_df['Mots'].apply(lambda x: set(x.split(',')))
        return classification_df
    except Exception as e:
        logger.warning(f"Impossible de charger la classification des mots depuis {classification_path}: {str(e)}")
        return pd.DataFrame(columns=['Theme', 'Mots'])

def clean_csv_file(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Nettoie et corrige un fichier CSV problématique, gérant les problèmes courants comme:
    - Guillemets non équilibrés
    - Sauts de ligne à l'intérieur des champs
    - Nombre incorrect de colonnes
    
    Args:
        input_path (str): Chemin du fichier CSV à nettoyer
        output_path (Optional[str]): Chemin où enregistrer le fichier nettoyé.
                                   Si None, utilise input_path + '_cleaned.csv'
    
    Returns:
        str: Chemin du fichier nettoyé
    """
    # Générer un ID unique pour cette opération
    op_id = str(uuid.uuid4())[:8]
    
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cleaned{ext}"
    
    logger.info(f"[{op_id}] Nettoyage du fichier CSV: {input_path} -> {output_path}")
    
    # Vérifier si le fichier existe
    if not os.path.exists(input_path):
        logger.error(f"[{op_id}] Fichier introuvable: {input_path}")
        raise FileNotFoundError(f"Le fichier {input_path} n'existe pas")
    
    # Lire le fichier ligne par ligne et corriger les problèmes
    with open(input_path, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_path, 'w', encoding='utf-8', newline='') as outfile:
         
        # Lire l'en-tête pour déterminer le nombre de colonnes
        header = infile.readline().strip()
        columns = header.split(',')
        num_columns = len(columns)
        
        logger.info(f"[{op_id}] En-tête détectée: {columns}")
        logger.info(f"[{op_id}] Nombre de colonnes attendues: {num_columns}")
        
        # Écrire l'en-tête dans le fichier de sortie
        outfile.write(header + '\n')
        
        # Variables pour suivre le processus
        line_count = 1  # Déjà lu la ligne d'en-tête
        fixed_count = 0
        buffer = ""
        
        # Traiter le reste du fichier
        for line in infile:
            line_count += 1
            
            # Si nous sommes en train de traiter une ligne incomplète
            if buffer:
                line = buffer + line
                buffer = ""
            
            # Compter les guillemets pour voir si nous avons un champ non fermé
            if line.count('"') % 2 != 0:
                # Guillemets non équilibrés, stocker dans le buffer et continuer
                buffer = line.rstrip('\n')
                continue
            
            # Essayer de diviser la ligne en colonnes
            # En utilisant le module csv pour gérer correctement les champs entre guillemets
            try:
                fields = list(csv.reader([line]))[0]
            except Exception as e:
                logger.warning(f"[{op_id}] Erreur de lecture CSV à la ligne {line_count}: {str(e)}")
                # Remplacer les caractères problématiques
                line = line.replace('\0', '')
                try:
                    fields = list(csv.reader([line]))[0]
                except Exception:
                    logger.error(f"[{op_id}] Impossible de corriger la ligne {line_count}, elle sera ignorée")
                    continue
            
            if len(fields) == num_columns:
                # Ligne correcte, l'écrire telle quelle
                outfile.write(line)
            else:
                fixed_count += 1
                if len(fields) < num_columns:
                    # Pas assez de colonnes, ajouter des champs vides
                    fields.extend([''] * (num_columns - len(fields)))
                    # Reconstruire la ligne avec le bon nombre de colonnes
                    corrected_line = ','.join([f'"{f}"' if ',' in f else f for f in fields])
                    outfile.write(corrected_line + '\n')
                else:
                    # Trop de colonnes, fusionner les colonnes excédentaires avec la dernière colonne attendue
                    corrected_fields = fields[:num_columns-1]
                    remaining = ','.join(fields[num_columns-1:])
                    corrected_fields.append(remaining)
                    # Reconstruire la ligne avec le bon nombre de colonnes
                    corrected_line = ','.join([f'"{f}"' if ',' in f else f for f in corrected_fields])
                    outfile.write(corrected_line + '\n')
    
    logger.info(f"[{op_id}] Nettoyage terminé: {line_count} lignes traitées, {fixed_count} lignes corrigées")
    
    # Vérifier si le fichier nettoyé peut être lu par pandas
    try:
        # Vérifier simplement que pandas peut ouvrir le fichier
        df = pd.read_csv(output_path, nrows=5)
        logger.info(f"[{op_id}] Le fichier nettoyé peut être lu avec pandas: {len(df)} lignes échantillonnées")
    except Exception as e:
        logger.error(f"[{op_id}] Le fichier nettoyé ne peut pas être lu avec pandas: {str(e)}")
        raise ValueError(f"Le fichier nettoyé ne peut pas être lu correctement: {str(e)}")
    
    return output_path

class DataIngestionPipeline:
    def __init__(self, data_path: str, experiment_name: str = INGESTION_EXPERIMENT_NAME, 
                 is_validation_set: bool = False, original_filename: str = None,
                 dataset_name: str = None):
        """
        Initialise le pipeline d'ingestion de données.
        
        Args:
            data_path (str): Chemin vers le fichier de données
            experiment_name (str): Nom de l'expérience MLflow
            is_validation_set (bool): Si True, les données seront taguées comme "jdd validation"
                                     sinon comme "jdd entrainement"
            original_filename (str): Nom du fichier original uploadé par l'utilisateur
            dataset_name (str): Nom personnalisé du dataset défini par l'utilisateur
        """
        # Génération d'un ID unique pour ce pipeline d'ingestion
        self.pipeline_id = str(uuid.uuid4())[:8]
        logger.info(f"[{self.pipeline_id}] Initialisation du pipeline d'ingestion - Fichier: {data_path}")
        
        self.data_path = data_path
        self.required_columns = REQUIRED_COLUMNS
        self.original_filename = original_filename or os.path.basename(data_path)
        
        # Définir un nom de dataset par défaut basé sur le type et le timestamp si non fourni
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_type = "Validation" if is_validation_set else "Entraînement"
        self.dataset_name = dataset_name or f"Jeu de données {dataset_type} {timestamp}"
        
        # Définition du type de dataset
        self.dataset_type = "jdd validation" if is_validation_set else "jdd entrainement"
        logger.info(f"[{self.pipeline_id}] Type de jeu de données: {self.dataset_type}")
        
        # Configuration MLflow
        logger.info(f"[{self.pipeline_id}] Configuration de l'expérience MLflow: {experiment_name}")
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        logger.debug(f"[{self.pipeline_id}] Expérience configurée: ID={self.experiment.experiment_id}")
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Vérifie que les données contiennent les colonnes requises.
        
        Args:
            data (pd.DataFrame): Données à valider
            
        Returns:
            bool: True si les données sont valides
        """
        logger.info(f"[{self.pipeline_id}] Validation des données - {len(data)} lignes")
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"[{self.pipeline_id}] Validation échouée - Colonnes manquantes: {missing_columns}")
            raise ValueError(f"Colonnes manquantes: {missing_columns}")
            
        logger.info(f"[{self.pipeline_id}] Validation réussie - Toutes les colonnes requises sont présentes")
        return True

    def get_data_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcule les statistiques des données.
        
        Args:
            data (pd.DataFrame): Données
            
        Returns:
            Dict: Statistiques des données
        """
        logger.info(f"[{self.pipeline_id}] Calcul des statistiques sur {len(data)} lignes")
        
        # Mesure du temps de calcul
        start_time = time.time()
        
        # Conversion des types numpy en types Python standard
        stats = {
            "n_rows": int(len(data)),
            "n_missing_avis": int(data["Avis"].isna().sum()),
            "n_missing_notes": int(data["Note"].isna().sum()),
            "avg_note": float(data["Note"].mean()),
            "min_note": int(data["Note"].min()),
            "max_note": int(data["Note"].max()),
            "avg_avis_length": float(data["Avis"].str.len().mean())
        }
        
        # Calcul du temps d'exécution
        execution_time = time.time() - start_time
        logger.info(f"[{self.pipeline_id}] Statistiques calculées en {execution_time:.3f}s")
        logger.debug(f"[{self.pipeline_id}] Statistiques: {stats}")
        
        return stats

    def load_data(self) -> pd.DataFrame:
        """
        Charge les données depuis le fichier source.
        Nettoie automatiquement le fichier CSV s'il y a des problèmes de format.
        
        Returns:
            pd.DataFrame: Données chargées
        """
        logger.info(f"[{self.pipeline_id}] Chargement des données depuis {self.data_path}")
        
        # Vérification de l'existence du fichier
        if not Path(self.data_path).exists():
            logger.error(f"[{self.pipeline_id}] Fichier introuvable: {self.data_path}")
            raise FileNotFoundError(f"Le fichier {self.data_path} n'existe pas")
        
        # Mesure du temps de chargement
        start_time = time.time()
        
        try:
            # Première tentative de lecture directe
            try:
                logger.info(f"[{self.pipeline_id}] Tentative de lecture directe du fichier CSV")
                data = pd.read_csv(self.data_path)
                load_time = time.time() - start_time
                logger.info(f"[{self.pipeline_id}] Données chargées en {load_time:.3f}s - {len(data)} lignes")
            except Exception as e:
                logger.warning(f"[{self.pipeline_id}] Erreur lors de la lecture du CSV: {str(e)}")
                
                # Vérifier si le nettoyage automatique est activé
                if AUTO_CLEAN_CSV:
                    logger.info(f"[{self.pipeline_id}] Tentative de nettoyage du fichier CSV problématique")
                    
                    # Essayer de nettoyer le fichier
                    cleaned_path = clean_csv_file(self.data_path)
                    
                    # Charger le fichier nettoyé
                    logger.info(f"[{self.pipeline_id}] Chargement du fichier nettoyé: {cleaned_path}")
                    data = pd.read_csv(cleaned_path)
                    load_time = time.time() - start_time
                    logger.info(f"[{self.pipeline_id}] Données chargées après nettoyage en {load_time:.3f}s - {len(data)} lignes")
                else:
                    logger.error(f"[{self.pipeline_id}] Nettoyage automatique désactivé, impossible de charger le fichier CSV problématique")
                    raise ValueError(f"Le fichier CSV est corrompu et le nettoyage automatique est désactivé: {str(e)}")
            
            # Validation des données
            self.validate_data(data)
            
            return data
        except Exception as e:
            logger.error(f"[{self.pipeline_id}] Erreur lors du chargement des données: {str(e)}", exc_info=True)
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prétraite les données avec des méthodes avancées de traitement de texte.
        
        Args:
            data (pd.DataFrame): Données brutes
            
        Returns:
            pd.DataFrame: Données prétraitées
        """
        logger.info(f"[{self.pipeline_id}] Début du prétraitement des données - {len(data)} lignes")
        
        # Mesure du temps de prétraitement
        start_time = time.time()
        
        try:
            processed_data = data.copy()
            
            # Nettoyage des Avis
            logger.debug(f"[{self.pipeline_id}] Nettoyage des avis")
            processed_data["Avis"] = processed_data["Avis"].astype(str).str.strip()
            processed_data["Avis"] = processed_data["Avis"].replace(r'^\s*$', np.nan, regex=True)
            
            # Validation des Notes
            logger.debug(f"[{self.pipeline_id}] Validation des notes")
            initial_note_count = processed_data["Note"].count()
            processed_data["Note"] = pd.to_numeric(processed_data["Note"], errors="coerce")
            converted_note_count = processed_data["Note"].count()
            
            if initial_note_count != converted_note_count:
                logger.warning(f"[{self.pipeline_id}] {initial_note_count - converted_note_count} notes non numériques ont été converties en NaN")
            
            # Suppression des lignes avec des valeurs manquantes
            logger.debug(f"[{self.pipeline_id}] Suppression des lignes avec valeurs manquantes")
            initial_rows = len(processed_data)
            processed_data = processed_data.dropna(subset=["Avis", "Note"])
            dropped_rows = initial_rows - len(processed_data)
            
            if dropped_rows > 0:
                logger.info(f"[{self.pipeline_id}] {dropped_rows} lignes supprimées pour valeurs manquantes ({dropped_rows/initial_rows:.2%})")
            
            # Conversion des types supplémentaires
            if 'Date' in processed_data.columns:
                processed_data['Date'] = pd.to_datetime(processed_data['Date'], errors='coerce')
                
            # Gestion des valeurs manquantes pour 'Réponse'
            if 'Réponse' in processed_data.columns:
                processed_data['Réponse'] = processed_data['Réponse'].replace(['nan', 'Pas de réponse'], None)
                
            # Encodage de la colonne 'Vérifié' si présente
            if 'Vérifié' in processed_data.columns:
                processed_data['Vérifié'] = processed_data['Vérifié'].apply(lambda x: True if x == 'Vérifié' else False)
            
            # Prétraitement avancé de texte (si les dépendances sont disponibles)
            if ADVANCED_PREPROCESSING_AVAILABLE:
                logger.info(f"[{self.pipeline_id}] Application du prétraitement avancé de texte")
                
                try:
                    # Charger spaCy et les ressources nécessaires
                    nlp = spacy.load("fr_core_news_sm")
                    stop_words = set(stopwords.words('french'))
                    
                    # Charger le vocabulaire français
                    french_vocab = load_french_vocabulary()
                    
                    if french_vocab:
                        logger.debug(f"[{self.pipeline_id}] Vocabulaire français chargé: {len(french_vocab)} mots")
                        
                        # Prétraitement des textes
                        logger.info(f"[{self.pipeline_id}] Prétraitement des textes...")
                        processed_data['Mots_importants'] = processed_data['Avis'].apply(
                            lambda x: preprocess_text(x, french_vocab, nlp, stop_words)
                        )
                        
                        # Prétraitement des réponses si présentes
                        if 'Réponse' in processed_data.columns:
                            processed_data['Mots_importants_reponse'] = processed_data['Réponse'].apply(
                                lambda x: preprocess_text(x, french_vocab, nlp, stop_words)
                            )
                            
                        # Détermination des thèmes
                        try:
                            logger.info(f"[{self.pipeline_id}] Détermination des thèmes...")
                            classification_df = load_classification_df()
                            
                            if not classification_df.empty:
                                processed_data['Themes_Avis'] = processed_data['Mots_importants'].apply(
                                    lambda x: determine_themes(x, classification_df)
                                )
                                
                                if 'Mots_importants_reponse' in processed_data.columns:
                                    processed_data['Themes_Réponse'] = processed_data['Mots_importants_reponse'].apply(
                                        lambda x: determine_themes(x, classification_df)
                                    )
                        except Exception as theme_error:
                            logger.warning(f"[{self.pipeline_id}] Erreur lors de la détermination des thèmes: {str(theme_error)}")
                    
                except Exception as preproc_error:
                    logger.warning(f"[{self.pipeline_id}] Erreur lors du prétraitement avancé: {str(preproc_error)}")
                    logger.info(f"[{self.pipeline_id}] Poursuite du traitement sans prétraitement avancé")
            else:
                logger.info(f"[{self.pipeline_id}] Prétraitement avancé non disponible, utilisation du prétraitement standard")
                
            # Création de la colonne sentiment
            processed_data['Sentiment'] = processed_data['Note'].apply(lambda x: 'Positif' if x > 3 else 'Négatif')
            
            # Log des statistiques de prétraitement
            stats = self.get_data_stats(processed_data)
            
            # Calcul du temps de prétraitement
            execution_time = time.time() - start_time
            logger.info(f"[{self.pipeline_id}] Prétraitement terminé en {execution_time:.3f}s - {len(processed_data)} lignes conservées")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"[{self.pipeline_id}] Erreur lors du prétraitement: {str(e)}", exc_info=True)
            raise
    


    def run_pipeline(self) -> pd.DataFrame:
        """
        Exécute le pipeline complet d'ingestion de données.
        
        Returns:
            pd.DataFrame: Données prétraitées
        """
        logger.info(f"[{self.pipeline_id}] Démarrage du pipeline d'ingestion")
        start_time = time.time()
        
        with mlflow.start_run(experiment_id=self.experiment.experiment_id) as run:
            logger.info(f"[{self.pipeline_id}] Run MLflow démarré: {run.info.run_id}")
            try:
                # Chargement des données
                logger.info(f"[{self.pipeline_id}] Étape 1: Chargement des données")
                data = self.load_data()
                initial_stats = self.get_data_stats(data)
                
                # Log des paramètres
                logger.debug(f"[{self.pipeline_id}] Enregistrement des paramètres dans MLflow")
                mlflow.log_param("data_path", self.data_path)
                mlflow.log_param("initial_rows", initial_stats["n_rows"])
                mlflow.log_param("pipeline_id", self.pipeline_id)
                mlflow.log_param("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                mlflow.log_param("original_filename", self.original_filename)
                mlflow.log_param("dataset_name", self.dataset_name)
                
                # Prétraitement
                logger.info(f"[{self.pipeline_id}] Étape 2: Prétraitement des données")
                processed_data = self.preprocess_data(data)
                
                final_stats = self.get_data_stats(processed_data)
                
                # Log des métriques
                logger.debug(f"[{self.pipeline_id}] Enregistrement des métriques dans MLflow")
                removed_rows = initial_stats["n_rows"] - final_stats["n_rows"]
                removed_pct = (removed_rows / initial_stats["n_rows"]) * 100 if initial_stats["n_rows"] > 0 else 0
                
                mlflow.log_metrics({
                    "final_rows": final_stats["n_rows"],
                    "removed_rows": removed_rows,
                    "removed_pct": removed_pct,
                    "avg_note": final_stats["avg_note"],
                    "avg_avis_length": final_stats["avg_avis_length"]
                })
                
                # Sauvegarde temporaire des données pour MLflow
                logger.debug(f"[{self.pipeline_id}] Sauvegarde temporaire des données pour MLflow")
                temp_input_path = f"temp_input_data_{self.pipeline_id}.csv"
                temp_output_path = f"temp_processed_data_{self.pipeline_id}.csv"
                
                data.to_csv(temp_input_path, index=False)
                processed_data.to_csv(temp_output_path, index=False)
                
                # Log des données dans MLflow
                logger.debug(f"[{self.pipeline_id}] Enregistrement des artifacts dans MLflow")
                mlflow.log_artifact(temp_input_path, "data_input")
                mlflow.log_artifact(temp_output_path, "data_processed")
                
                # Tag du jeu de données
                logger.info(f"[{self.pipeline_id}] Application du tag '{self.dataset_type}' au run")
                client = mlflow.tracking.MlflowClient()
                client.set_tag(run.info.run_id, "dataset_type", self.dataset_type)
                client.set_tag(run.info.run_id, "dataset_rows", str(final_stats["n_rows"]))
                client.set_tag(run.info.run_id, "dataset_version", datetime.now().strftime("%Y%m%d_%H%M%S"))
                
                # Log des distributions sous forme de visualisations
                logger.info(f"[{self.pipeline_id}] Étape 3: Génération de visualisations")
                import matplotlib.pyplot as plt
                
                # Distribution des notes
                logger.debug(f"[{self.pipeline_id}] Création de la distribution des notes")
                plt.figure(figsize=(10, 6))
                processed_data['Note'].hist()
                plt.title('Distribution des Notes')
                plt.xlabel('Note')
                plt.ylabel('Fréquence')
                notes_viz_path = f"notes_distribution_{self.pipeline_id}.png"
                plt.savefig(notes_viz_path)
                mlflow.log_artifact(notes_viz_path, "visualizations")
                plt.close()
                
                # Distribution de la longueur des avis
                logger.debug(f"[{self.pipeline_id}] Création de la distribution des longueurs d'avis")
                plt.figure(figsize=(10, 6))
                processed_data['Avis'].str.len().hist()
                plt.title('Distribution de la longueur des avis')
                plt.xlabel('Longueur du texte')
                plt.ylabel('Fréquence')
                avis_viz_path = f"avis_length_distribution_{self.pipeline_id}.png"
                plt.savefig(avis_viz_path)
                mlflow.log_artifact(avis_viz_path, "visualizations")
                plt.close()
                
                # Nettoyage des fichiers temporaires
                logger.debug(f"[{self.pipeline_id}] Nettoyage des fichiers temporaires")
                try:
                    os.remove(temp_input_path)
                    os.remove(temp_output_path)
                    os.remove(notes_viz_path)
                    os.remove(avis_viz_path)
                except Exception as e:
                    logger.warning(f"[{self.pipeline_id}] Erreur lors du nettoyage des fichiers temporaires: {str(e)}")
                
                # Calcul du temps total d'exécution
                total_time = time.time() - start_time
                logger.info(f"[{self.pipeline_id}] Pipeline d'ingestion terminé avec succès en {total_time:.3f}s")
                mlflow.log_metric("pipeline_execution_time", total_time)
                
                return processed_data
                
            except Exception as e:
                logger.error(f"[{self.pipeline_id}] Erreur pendant l'ingestion: {str(e)}", exc_info=True)
                mlflow.log_param("error", str(e))
                mlflow.log_param("error_type", type(e).__name__)
                raise
