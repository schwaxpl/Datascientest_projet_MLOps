"""
Test script pour vérifier le fonctionnement des modifications du dataset_name dans l'API Gateway.

Ce script teste:
1. Upload d'un dataset avec nom personnalisé
2. Vérification que le dataset_name est correctement enregistré
3. Balance d'un dataset avec un nom personnalisé
4. Vérification que le dataset_name est propagé
"""

import os
import requests
import json
import time
from pprint import pprint

# Configuration
API_URL = "http://localhost:8000"
USERNAME = "user"
PASSWORD = "userpassword"
TEST_FILE = "data/example_avis.csv"

# Fonction pour obtenir un token d'authentification
def get_auth_token():
    auth_data = {
        "username": USERNAME,
        "password": PASSWORD
    }
    response = requests.post(
        f"{API_URL}/token",
        data=auth_data
    )
    return response.json()["access_token"]

# Fonction principale de test
def test_dataset_name_functionality():
    # 1. Obtention du token
    print("1. Obtention du token d'authentification...")
    token = get_auth_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Upload d'un dataset avec un nom personnalisé
    print("\n2. Upload d'un dataset avec un nom personnalisé...")
    test_dataset_name = "Mon Dataset Test Custom"
    
    if not os.path.exists(TEST_FILE):
        print(f"Erreur: Le fichier test {TEST_FILE} n'existe pas.")
        return
    
    with open(TEST_FILE, "rb") as f:
        files = {"file": f}
        data = {"dataset_name": test_dataset_name}
        response = requests.post(
            f"{API_URL}/data/upload",
            headers=headers,
            files=files,
            data=data
        )
    
    if response.status_code != 200:
        print(f"Erreur lors de l'upload: {response.status_code}")
        print(response.text)
        return
    
    upload_result = response.json()
    dataset_id = upload_result.get("id")
    print(f"Dataset uploadé avec succès. ID: {dataset_id}")
    
    # 3. Vérification que le dataset_name est correctement enregistré
    print("\n3. Vérification du dataset_name...")
    response = requests.get(
        f"{API_URL}/data/datasets/{dataset_id}",
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"Erreur lors de la récupération du dataset: {response.status_code}")
        print(response.text)
        return
    
    dataset_info = response.json()
    print("Informations du dataset:")
    pprint(dataset_info)
    
    if dataset_info.get("dataset_name") == test_dataset_name:
        print("✅ Test réussi: Le nom du dataset est correctement enregistré.")
    else:
        print(f"❌ Test échoué: Le nom du dataset ({dataset_info.get('dataset_name')}) ne correspond pas à la valeur attendue ({test_dataset_name}).")
    
    # 4. Balance d'un dataset avec un nom personnalisé
    print("\n4. Balance d'un dataset avec un nom personnalisé...")
    balanced_dataset_name = "Mon Dataset Équilibré Custom"
    balance_data = {
        "dataset_id": dataset_id,
        "strategy": "hybrid",
        "target_ratio": 0.5,
        "random_seed": 42,
        "dataset_name": balanced_dataset_name
    }
    
    response = requests.post(
        f"{API_URL}/data/datasets/balance",
        headers=headers,
        data=balance_data
    )
    
    if response.status_code != 200:
        print(f"Erreur lors de l'équilibrage: {response.status_code}")
        print(response.text)
        return
    
    balance_result = response.json()
    balanced_dataset_id = balance_result.get("balanced_dataset_id")
    print(f"Dataset équilibré avec succès. ID: {balanced_dataset_id}")
    
    # 5. Vérification que le dataset_name est propagé
    print("\n5. Vérification du dataset_name après équilibrage...")
    response = requests.get(
        f"{API_URL}/data/datasets/{balanced_dataset_id}",
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"Erreur lors de la récupération du dataset équilibré: {response.status_code}")
        print(response.text)
        return
    
    balanced_dataset_info = response.json()
    print("Informations du dataset équilibré:")
    pprint(balanced_dataset_info)
    
    if balanced_dataset_info.get("dataset_name") == balanced_dataset_name:
        print("✅ Test réussi: Le nom du dataset équilibré est correctement enregistré.")
    else:
        print(f"❌ Test échoué: Le nom du dataset équilibré ({balanced_dataset_info.get('dataset_name')}) ne correspond pas à la valeur attendue ({balanced_dataset_name}).")
    
    print("\n✨ Tests terminés.")

if __name__ == "__main__":
    test_dataset_name_functionality()
