"""
Package commun pour les microservices.
Contient des utilitaires et configurations partagés.
"""

from microservices.common.config import *
from microservices.common.logger_config import init_logging, get_logger
from microservices.common.utils import (
    get_mlflow_client,
    load_model_from_registry,
    get_latest_model_version,
    get_vectorizer_from_run,
    call_service,
    create_access_token,
    get_current_user,
    get_current_active_user,
    check_admin_role,
    measure_time
)
