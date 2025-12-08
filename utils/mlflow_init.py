import mlflow
from utils.config_loader import config
from utils.logging import logger

def init_mlflow():
    mlflow_cfg = config.get("mlflow", {}) or {}
    tracking_uri = mlflow_cfg.get("tracking_uri") or __import__('os').environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.warning("MLFLOW_TRACKING_URI not set; MLflow will not log.")
        return
    try:
        mlflow.set_tracking_uri(tracking_uri)
        default_experiment = mlflow_cfg.get("default_experiment")
        mlflow.set_experiment(default_experiment)
        logger.info("Configured MLflow tracking URI=%s experiment=%s", tracking_uri, default_experiment)
    except Exception:
        logger.exception("Failed to initialize MLflow tracking")
