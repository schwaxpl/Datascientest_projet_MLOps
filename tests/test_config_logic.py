"""
Tests pour la logique de configuration et paramétrage - Logique métier pure
"""
import pytest
import os
from typing import Dict, List, Any, Optional


class TestConfigurationLogic:
    """Tests pour la logique de configuration"""

    def test_default_configuration_logic(self):
        """Test de la logique de configuration par défaut"""
        def get_default_config() -> Dict:
            """Retourne la configuration par défaut"""
            return {
                'model': {
                    'type': 'sentiment_analysis',
                    'algorithm': 'tfidf_logistic',
                    'min_accuracy': 0.8,
                    'max_features': 10000
                },
                'data': {
                    'train_split': 0.8,
                    'val_split': 0.1,
                    'test_split': 0.1,
                    'min_samples': 100
                },
                'preprocessing': {
                    'lowercase': True,
                    'remove_punctuation': True,
                    'remove_stopwords': True,
                    'min_word_length': 2
                },
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            }
        
        config = get_default_config()
        
        # Vérifications de structure
        assert 'model' in config
        assert 'data' in config
        assert 'preprocessing' in config
        assert 'logging' in config
        
        # Vérifications de valeurs
        assert config['model']['min_accuracy'] == 0.8
        assert config['data']['train_split'] + config['data']['val_split'] + config['data']['test_split'] == 1.0
        assert config['preprocessing']['lowercase'] == True

    def test_configuration_validation_logic(self):
        """Test de validation de configuration"""
        def validate_config(config: Dict) -> List[str]:
            """Valide une configuration et retourne les erreurs"""
            errors = []
            
            # Validation des splits de données
            if 'data' in config:
                splits = config['data']
                total_split = splits.get('train_split', 0) + splits.get('val_split', 0) + splits.get('test_split', 0)
                if abs(total_split - 1.0) > 0.001:
                    errors.append(f"Data splits must sum to 1.0, got {total_split}")
                
                if splits.get('min_samples', 0) < 10:
                    errors.append("min_samples must be at least 10")
            
            # Validation du modèle
            if 'model' in config:
                model_config = config['model']
                if model_config.get('min_accuracy', 0) < 0.5:
                    errors.append("min_accuracy must be at least 0.5")
                
                if model_config.get('max_features', 0) < 100:
                    errors.append("max_features must be at least 100")
            
            return errors
        
        # Configuration valide
        valid_config = {
            'data': {'train_split': 0.8, 'val_split': 0.1, 'test_split': 0.1, 'min_samples': 100},
            'model': {'min_accuracy': 0.8, 'max_features': 5000}
        }
        errors = validate_config(valid_config)
        assert len(errors) == 0
        
        # Configuration invalide
        invalid_config = {
            'data': {'train_split': 0.9, 'val_split': 0.1, 'test_split': 0.1, 'min_samples': 5},
            'model': {'min_accuracy': 0.3, 'max_features': 50}
        }
        errors = validate_config(invalid_config)
        assert len(errors) == 4  # 4 erreurs attendues

    def test_environment_specific_config(self):
        """Test de configuration spécifique à l'environnement"""
        def get_env_config(environment: str) -> Dict:
            """Retourne la configuration pour un environnement spécifique"""
            base_config = {
                'debug': False,
                'log_level': 'INFO',
                'batch_size': 32,
                'timeout': 30
            }
            
            if environment == 'development':
                base_config.update({
                    'debug': True,
                    'log_level': 'DEBUG',
                    'batch_size': 16
                })
            elif environment == 'testing':
                base_config.update({
                    'debug': True,
                    'log_level': 'WARNING',
                    'batch_size': 8,
                    'timeout': 10
                })
            elif environment == 'production':
                base_config.update({
                    'debug': False,
                    'log_level': 'ERROR',
                    'batch_size': 64,
                    'timeout': 60
                })
            
            return base_config
        
        # Test des différents environnements
        dev_config = get_env_config('development')
        assert dev_config['debug'] == True
        assert dev_config['log_level'] == 'DEBUG'
        
        prod_config = get_env_config('production')
        assert prod_config['debug'] == False
        assert prod_config['log_level'] == 'ERROR'
        assert prod_config['timeout'] == 60

    def test_parameter_override_logic(self):
        """Test de logique de surcharge de paramètres"""
        def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
            """Fusionne deux configurations avec priorité à override_config"""
            result = base_config.copy()
            
            for key, value in override_config.items():
                if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                    result[key] = merge_configs(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        base = {
            'model': {'type': 'logistic', 'accuracy': 0.8},
            'data': {'split': 0.8},
            'debug': False
        }
        
        override = {
            'model': {'accuracy': 0.9},  # Override partiel
            'data': {'split': 0.7, 'validation': True},  # Override + ajout
            'new_param': 'value'  # Nouveau paramètre
        }
        
        merged = merge_configs(base, override)
        
        assert merged['model']['type'] == 'logistic'  # Conservé de base
        assert merged['model']['accuracy'] == 0.9     # Surchargé
        assert merged['data']['split'] == 0.7         # Surchargé
        assert merged['data']['validation'] == True   # Ajouté
        assert merged['debug'] == False               # Conservé de base
        assert merged['new_param'] == 'value'         # Ajouté


class TestParameterValidation:
    """Tests pour la validation des paramètres"""

    def test_numeric_parameter_validation(self):
        """Test de validation des paramètres numériques"""
        def validate_numeric_param(value: Any, param_name: str, min_val: Optional[float] = None, 
                                 max_val: Optional[float] = None) -> List[str]:
            """Valide un paramètre numérique"""
            errors = []
            
            # Vérifier le type
            if not isinstance(value, (int, float)):
                errors.append(f"{param_name} must be numeric, got {type(value).__name__}")
                return errors
            
            # Vérifier les bornes
            if min_val is not None and value < min_val:
                errors.append(f"{param_name} must be >= {min_val}, got {value}")
            
            if max_val is not None and value > max_val:
                errors.append(f"{param_name} must be <= {max_val}, got {value}")
            
            return errors
        
        # Tests valides
        assert len(validate_numeric_param(0.8, "accuracy", 0.0, 1.0)) == 0
        assert len(validate_numeric_param(100, "batch_size", 1)) == 0
        
        # Tests invalides
        errors = validate_numeric_param("not_a_number", "accuracy")
        assert len(errors) == 1
        assert "must be numeric" in errors[0]
        
        errors = validate_numeric_param(-0.5, "accuracy", 0.0, 1.0)
        assert len(errors) == 1
        assert "must be >=" in errors[0]

    def test_string_parameter_validation(self):
        """Test de validation des paramètres string"""
        def validate_string_param(value: Any, param_name: str, 
                                allowed_values: Optional[List[str]] = None,
                                min_length: Optional[int] = None) -> List[str]:
            """Valide un paramètre string"""
            errors = []
            
            # Vérifier le type
            if not isinstance(value, str):
                errors.append(f"{param_name} must be a string, got {type(value).__name__}")
                return errors
            
            # Vérifier la longueur
            if min_length is not None and len(value) < min_length:
                errors.append(f"{param_name} must be at least {min_length} characters, got {len(value)}")
            
            # Vérifier les valeurs autorisées
            if allowed_values is not None and value not in allowed_values:
                errors.append(f"{param_name} must be one of {allowed_values}, got '{value}'")
            
            return errors
        
        # Tests valides
        assert len(validate_string_param("logistic", "algorithm", ["logistic", "svm"])) == 0
        assert len(validate_string_param("test_name", "name", min_length=5)) == 0
        
        # Tests invalides
        errors = validate_string_param(123, "algorithm")
        assert len(errors) == 1
        assert "must be a string" in errors[0]
        
        errors = validate_string_param("invalid", "algorithm", ["logistic", "svm"])
        assert len(errors) == 1
        assert "must be one of" in errors[0]

    def test_boolean_parameter_validation(self):
        """Test de validation des paramètres booléens"""
        def validate_boolean_param(value: Any, param_name: str) -> List[str]:
            """Valide un paramètre booléen"""
            errors = []
            
            if not isinstance(value, bool):
                errors.append(f"{param_name} must be a boolean, got {type(value).__name__}")
            
            return errors
        
        # Tests valides
        assert len(validate_boolean_param(True, "debug")) == 0
        assert len(validate_boolean_param(False, "verbose")) == 0
        
        # Tests invalides
        errors = validate_boolean_param("true", "debug")
        assert len(errors) == 1
        assert "must be a boolean" in errors[0]

    def test_list_parameter_validation(self):
        """Test de validation des paramètres de type liste"""
        def validate_list_param(value: Any, param_name: str, 
                               min_length: Optional[int] = None,
                               max_length: Optional[int] = None,
                               item_type: Optional[type] = None) -> List[str]:
            """Valide un paramètre de type liste"""
            errors = []
            
            # Vérifier le type
            if not isinstance(value, list):
                errors.append(f"{param_name} must be a list, got {type(value).__name__}")
                return errors
            
            # Vérifier la longueur
            if min_length is not None and len(value) < min_length:
                errors.append(f"{param_name} must have at least {min_length} items, got {len(value)}")
            
            if max_length is not None and len(value) > max_length:
                errors.append(f"{param_name} must have at most {max_length} items, got {len(value)}")
            
            # Vérifier le type des éléments
            if item_type is not None:
                for i, item in enumerate(value):
                    if not isinstance(item, item_type):
                        errors.append(f"{param_name}[{i}] must be {item_type.__name__}, got {type(item).__name__}")
            
            return errors
        
        # Tests valides
        assert len(validate_list_param([1, 2, 3], "features", min_length=1, item_type=int)) == 0
        assert len(validate_list_param(["a", "b"], "columns", max_length=5, item_type=str)) == 0
        
        # Tests invalides
        errors = validate_list_param("not_a_list", "features")
        assert len(errors) == 1
        assert "must be a list" in errors[0]
        
        errors = validate_list_param([1, "2", 3], "features", item_type=int)
        assert len(errors) == 1
        assert "must be int" in errors[0]


class TestConfigurationUtils:
    """Tests pour les utilitaires de configuration"""

    def test_config_path_resolution(self):
        """Test de résolution des chemins de configuration"""
        def resolve_config_path(relative_path: str, base_dir: str = "/app") -> str:
            """Résout un chemin de configuration relatif"""
            if relative_path.startswith('/'):
                return relative_path
            return f"{base_dir.rstrip('/')}/{relative_path}"
        
        # Tests de résolution
        assert resolve_config_path("config.yml") == "/app/config.yml"
        assert resolve_config_path("data/train.csv") == "/app/data/train.csv"
        assert resolve_config_path("/absolute/path.txt") == "/absolute/path.txt"
        assert resolve_config_path("config.yml", "/custom") == "/custom/config.yml"

    def test_config_template_generation(self):
        """Test de génération de template de configuration"""
        def generate_config_template(project_type: str) -> Dict:
            """Génère un template de configuration selon le type de projet"""
            base_template = {
                'project_name': f'{project_type}_project',
                'version': '1.0.0',
                'logging': {'level': 'INFO'}
            }
            
            if project_type == 'sentiment_analysis':
                base_template.update({
                    'model': {
                        'type': 'text_classification',
                        'algorithm': 'tfidf_logistic',
                        'min_accuracy': 0.8
                    },
                    'preprocessing': {
                        'lowercase': True,
                        'remove_stopwords': True
                    }
                })
            elif project_type == 'regression':
                base_template.update({
                    'model': {
                        'type': 'regression',
                        'algorithm': 'linear_regression',
                        'min_r2': 0.7
                    },
                    'features': {
                        'normalize': True,
                        'scale': True
                    }
                })
            
            return base_template
        
        # Test template sentiment analysis
        sentiment_config = generate_config_template('sentiment_analysis')
        assert sentiment_config['model']['type'] == 'text_classification'
        assert 'preprocessing' in sentiment_config
        
        # Test template regression
        regression_config = generate_config_template('regression')
        assert regression_config['model']['type'] == 'regression'
        assert 'features' in regression_config

    def test_config_comparison_logic(self):
        """Test de logique de comparaison de configurations"""
        def compare_configs(config1: Dict, config2: Dict) -> Dict:
            """Compare deux configurations et retourne les différences"""
            differences = {
                'added': {},
                'removed': {},
                'changed': {}
            }
            
            # Fonction récursive pour comparer
            def _compare_recursive(d1, d2, path=""):
                for key in d2:
                    current_path = f"{path}.{key}" if path else key
                    if key not in d1:
                        differences['added'][current_path] = d2[key]
                    elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                        _compare_recursive(d1[key], d2[key], current_path)
                    elif d1[key] != d2[key]:
                        differences['changed'][current_path] = {'old': d1[key], 'new': d2[key]}
                
                for key in d1:
                    current_path = f"{path}.{key}" if path else key
                    if key not in d2:
                        differences['removed'][current_path] = d1[key]
            
            _compare_recursive(config1, config2)
            return differences
        
        config1 = {
            'model': {'type': 'logistic', 'accuracy': 0.8},
            'data': {'split': 0.8},
            'old_param': 'value'
        }
        
        config2 = {
            'model': {'type': 'svm', 'accuracy': 0.8},
            'data': {'split': 0.7},
            'new_param': 'new_value'
        }
        
        diff = compare_configs(config1, config2)
        
        assert 'model.type' in diff['changed']
        assert 'data.split' in diff['changed']
        assert 'new_param' in diff['added']
        assert 'old_param' in diff['removed']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
