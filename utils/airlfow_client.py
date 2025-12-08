import requests
from typing import Any, Dict, Optional
from utils.config_loader import config
from utils.logging import logger

_af_cfg = config.get("airflow", {}) or {}
AIRFLOW_BASE =  _af_cfg.get("base_url", "http://localhost:8080")
AUTH_TYPE =  _af_cfg.get("auth", {}).get("type", "basic")
AIRFLOW_USER = _af_cfg.get("auth", {}).get("username")
AIRFLOW_PASS = _af_cfg.get("auth", {}).get("password")
AIRFLOW_TOKEN = _af_cfg.get("auth", {}).get("token")
DEFAULT_TIMEOUT = int(_af_cfg.get("timeout_seconds", 8))
DEFAULT_DAG_ID = _af_cfg.get("dag", {}).get("id", "evaluate_assignment")
DEFAULT_CONF_TEMPLATE = _af_cfg.get("dag", {}).get("conf_template", {}) or {}

def _build_headers_and_auth() -> (Dict[str,str], Optional[tuple]):
    headers = {"Content-Type": "application/json"}
    auth = None
    if AUTH_TYPE == "token":
        token = AIRFLOW_TOKEN
        if not token:
            raise RuntimeError("Airflow token auth selected but AIRFLOW_TOKEN missing")
        headers["Authorization"] = f"Bearer {token}"
    elif AUTH_TYPE == "basic":
        if not AIRFLOW_USER or not AIRFLOW_PASS:
            logger.warning("Airflow basic auth selected but AIRFLOW_USER/AIRFLOW_PASS missing; request may fail")
        auth = (AIRFLOW_USER, AIRFLOW_PASS)
    return headers, auth

def trigger_dag(dag_id: Optional[str] = None, conf: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
    dag_id = dag_id or DEFAULT_DAG_ID
    timeout = timeout or DEFAULT_TIMEOUT
    conf_payload = {}
    conf_payload.update(DEFAULT_CONF_TEMPLATE)
    if conf:
        conf_payload.update(conf)
    url = f"{AIRFLOW_BASE.rstrip('/')}/api/v1/dags/{dag_id}/dagRuns"
    headers, auth = _build_headers_and_auth()
    payload = {"conf": conf_payload}
    logger.info("airflow_client: triggering dag=%s url=%s conf_keys=%s", dag_id, url, list(conf_payload.keys()))
    try:
        resp = requests.post(url, json=payload, headers=headers, auth=auth, timeout=timeout)
        resp.raise_for_status()
        logger.info("airflow_client: triggered dag=%s status=%s", dag_id, resp.status_code)
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text}
    except Exception as exc:
        logger.exception("airflow_client: failed to trigger dag=%s: %s", dag_id, exc)
        raise
