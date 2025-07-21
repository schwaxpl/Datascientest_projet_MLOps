import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import spacy
from imblearn.over_sampling import RandomOverSampler

def preprocess_text(text, french_vocab, nlp, stop_words):
    if pd.isnull(text):
        return ""
    # Supprimer les caractères spéciaux et mettre en minuscule
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Supprimer les mots qui ne sont pas dans le dictionnaire français
    text = " ".join([word for word in text.split() if word in french_vocab])
    
    # Tokenizer, supprimer les stop words et appliquer la lemmatisation avec spaCy
    doc = nlp(text)
    words = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct and not token.is_space]
    return " ".join(words)

def determine_themes(text, classification_df, max_themes=5):
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

def process_dataframe(df_path, output_path=None, balance_method=None):
    """
    Process the dataframe with all preprocessing and feature engineering steps
    
    Parameters:
    -----------
    df_path : str
        Path to the input CSV file
    output_path : str, optional
        Path where to save the processed DataFrame
    balance_method : str, optional
        Method to balance classes: 'undersample' or 'oversample'
    """
    # Charger les données
    df = pd.read_csv(df_path)
    
    # Conversion des types
    df['Note'] = pd.to_numeric(df['Note'], errors='coerce').fillna(0).astype(int)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Gestion des valeurs manquantes
    df['Réponse'] = df['Réponse'].replace(['nan', 'Pas de réponse'], None)
    df['Avis'] = df['Avis'].replace(['nan', "Pas de texte d'avis"], None)
    
    # Encodage de la colonne 'Vérifié'
    df['Vérifié'] = df['Vérifié'].apply(lambda x: True if x == 'Vérifié' else False)
    
    # Feature Engineering
    # Charger spaCy et les ressources nécessaires
    nlp = spacy.load("fr_core_news_sm")
    stop_words = set(stopwords.words('french'))
    
    # Charger le vocabulaire français
    with open('liste_fr.txt', 'r', encoding='utf-8') as file:
        french_vocab = set(word.strip() for word in file.readlines())
    
    # Appliquer le preprocessing sur les textes
    print("Application du preprocessing sur les textes...")
    df['Mots_importants'] = df['Avis'].apply(lambda x: preprocess_text(x, french_vocab, nlp, stop_words))
    df['Mots_importants_reponse'] = df['Réponse'].apply(lambda x: preprocess_text(x, french_vocab, nlp, stop_words))
    
    # Détermination des thèmes
    print("Détermination des thèmes...")
    classification_df = pd.read_csv('Classification_mots.csv')
    classification_df['Mots'] = classification_df['Mots'].apply(lambda x: x.split(','))
    
    df['Themes_Avis'] = df['Mots_importants'].apply(lambda x: determine_themes(x, classification_df))
    df['Themes_Réponse'] = df['Mots_importants_reponse'].apply(lambda x: determine_themes(x, classification_df))
    
    # Création de la colonne sentiment
    df['Sentiment'] = df['Note'].apply(lambda x: 'Positif' if x > 3 else 'Négatif')
    
    # Équilibrage des classes si demandé
    if balance_method == 'undersample':
        print("Application du sous-échantillonnage...")
        min_count = df['Note'].value_counts().min()
        df = df.groupby('Note').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
    
    elif balance_method == 'oversample':
        print("Application du sur-échantillonnage...")
        try:
            ros = RandomOverSampler(random_state=42)
            X = df.drop(columns=['Note'])
            y = df['Note'].astype(str)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            y_resampled = y_resampled.astype(int)
            df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                          pd.DataFrame(y_resampled, columns=['Note'])], axis=1)
        except Exception as e:
            print(f"Erreur lors du sur-échantillonnage: {e}")
    
    # Sauvegarder si un chemin de sortie est spécifié
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"DataFrame sauvegardé dans {output_path}")
    
    return df

if __name__ == "__main__":
    # Exemple d'utilisation
    input_path = "data/TrustPilot_400.csv"  # Ajustez selon votre structure
    output_path = "data/processed_data.csv"  # Ajustez selon vos besoins
    
    # Vous pouvez choisir 'undersample', 'oversample' ou None
    processed_df = process_dataframe(input_path, output_path, balance_method='undersample')
    print("Traitement terminé.")
