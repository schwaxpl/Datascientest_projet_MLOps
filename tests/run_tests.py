"""
Script d'exécution de tous les tests.
Exécutez ce script pour lancer tous les tests unitaires du projet.
"""

import pytest
import sys
import os

def run_tests():
    """Exécute tous les tests unitaires du projet"""
    print("Démarrage des tests unitaires du projet MLOps...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    # Chemin des tests
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    # Arguments de base pour pytest
    pytest_args = [
        tests_dir,
        "-v",  # Mode verbose
    ]
    
    # Vérifie si pytest-cov est installé
    try:
        import pytest_cov
        # Si pytest-cov est installé, ajoute les arguments de couverture
        pytest_args.extend([
            "--cov=src",  # Couverture de code pour le répertoire src
            "--cov-report=term",  # Affichage de la couverture dans le terminal
            "--cov-report=html:coverage_reports",  # Génération d'un rapport HTML
        ])
        print("Génération du rapport de couverture activée")
    except ImportError:
        print("⚠️ Le module pytest-cov n'est pas installé. La couverture de code ne sera pas calculée.")
        print("   Pour l'installer, exécutez: pip install pytest-cov")
    
    # Exécution des tests avec pytest
    exit_code = pytest.main(pytest_args)
    
    # Si tous les tests passent, afficher un message de succès
    if exit_code == 0:
        print("\n✅ Tous les tests ont réussi!")
        print("Rapport de couverture généré dans le dossier 'coverage_reports'")
    else:
        print("\n❌ Certains tests ont échoué. Veuillez vérifier les erreurs ci-dessus.")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(run_tests())
