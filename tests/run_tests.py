"""
Script pour exécuter tous les tests
Usage: python -m pytest tests/ -v
ou: python tests/run_tests.py
"""

import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    """Exécute la suite de tests complète"""
    
    # Ajouter le répertoire racine au path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    
    print("🧪 Lancement de la suite de tests MLOps")
    print("=" * 50)
    
    # Vérifier que pytest est installé
    try:
        import pytest
    except ImportError:
        print("❌ pytest n'est pas installé. Installation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
        import pytest
    
    # Configuration des arguments pytest
    pytest_args = [
        str(Path(__file__).parent),  # Répertoire des tests
        "-v",  # Verbose
        "--tb=short",  # Traceback court
        "-x",  # Arrêter au premier échec
        "--disable-warnings",  # Supprimer les warnings
    ]
    
    print(f"📂 Répertoire de tests: {Path(__file__).parent}")
    print(f"🚀 Arguments pytest: {' '.join(pytest_args)}")
    print()
    
    # Exécuter les tests
    try:
        exit_code = pytest.main(pytest_args)
        
        if exit_code == 0:
            print("\n✅ Tous les tests sont passés avec succès !")
        else:
            print(f"\n❌ Tests échoués (code: {exit_code})")
            
        return exit_code
        
    except Exception as e:
        print(f"\n💥 Erreur lors de l'exécution des tests: {e}")
        return 1

def run_specific_test_module(module_name):
    """Exécute un module de test spécifique"""
    
    test_modules = {
        'preprocessing': 'test_preprocessing.py',
        'prediction': 'test_prediction.py',
        'utils': 'test_utils.py',
        'api': 'test_api_integration.py',
        'validation': 'test_model_validation.py'
    }
    
    if module_name not in test_modules:
        print(f"❌ Module '{module_name}' non trouvé.")
        print(f"Modules disponibles: {', '.join(test_modules.keys())}")
        return 1
    
    test_file = Path(__file__).parent / test_modules[module_name]
    
    if not test_file.exists():
        print(f"❌ Fichier de test '{test_file}' non trouvé.")
        return 1
    
    print(f"🧪 Exécution des tests: {module_name}")
    print("=" * 40)
    
    try:
        import pytest
        exit_code = pytest.main([str(test_file), "-v", "--tb=short"])
        return exit_code
    except Exception as e:
        print(f"💥 Erreur: {e}")
        return 1

def show_test_summary():
    """Affiche un résumé des tests disponibles"""
    test_dir = Path(__file__).parent
    
    print("📋 Tests disponibles dans le projet MLOps:")
    print("=" * 45)
    
    test_files = list(test_dir.glob("test_*.py"))
    
    for test_file in sorted(test_files):
        print(f"📄 {test_file.name}")
        
        # Lire la première ligne de docstring pour la description
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:10]:  # Chercher dans les 10 premières lignes
                    if line.strip().startswith('"""') and len(line.strip()) > 3:
                        desc = line.strip()[3:].strip()
                        if desc:
                            print(f"   📝 {desc}")
                        break
                    elif line.strip() and not line.strip().startswith('#'):
                        break
        except Exception:
            pass
        print()
    
    print("🚀 Commandes disponibles:")
    print("   python tests/run_tests.py                    # Tous les tests")
    print("   python tests/run_tests.py preprocessing      # Tests preprocessing")
    print("   python tests/run_tests.py prediction         # Tests prédiction")
    print("   python tests/run_tests.py utils              # Tests utilitaires")
    print("   python tests/run_tests.py api                # Tests APIs")
    print("   python tests/run_tests.py validation         # Tests validation")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "summary":
            show_test_summary()
        elif command in ['preprocessing', 'prediction', 'utils', 'api', 'validation']:
            sys.exit(run_specific_test_module(command))
        else:
            print(f"❌ Commande inconnue: {command}")
            show_test_summary()
            sys.exit(1)
    else:
        sys.exit(run_tests())
