# app.py
from fastapi import FastAPI, Query, HTTPException
import threading
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from utils.config_loader import config
from utils.mongo_client import db
from utils.logging import logger
from module.scanner_db_runner import scanner_loop, scan_and_process_once

app = FastAPI(title="Assignment Evaluator (modular)")

# Globals for scanner thread
_SCANNER_THREAD: Optional[threading.Thread] = None
_SCANNER_STOP_EVENT: Optional[threading.Event] = None


@app.on_event("startup")
def startup_event():
    logger.info("app.startup: starting Assignment Evaluator app")
    # Start the DB-scanner background thread if configured
    if config.get("scanner", {}).get("start_in_app", True):
        global _SCANNER_THREAD, _SCANNER_STOP_EVENT
        _SCANNER_STOP_EVENT = threading.Event()
        _SCANNER_THREAD = threading.Thread(
            target=scanner_loop,
            args=(_SCANNER_STOP_EVENT,),
            daemon=True
        )
        _SCANNER_THREAD.start()
        logger.info("app.startup: scanner loop started in background thread")


@app.on_event("shutdown")
def shutdown_event():
    logger.info("app.shutdown: shutting down Assignment Evaluator app")
    global _SCANNER_THREAD, _SCANNER_STOP_EVENT
    if _SCANNER_STOP_EVENT:
        _SCANNER_STOP_EVENT.set()
        logger.info("app.shutdown: signalled scanner to stop")
    if _SCANNER_THREAD:
        _SCANNER_THREAD.join(timeout=5)
        logger.info("app.shutdown: scanner thread join attempted")


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        _ = db.list_collection_names()
        return {"status": "ok", "mongo": True}
    except Exception:
        logger.exception("health: mongo check failed")
        return {"status": "degraded", "mongo": False}


@app.post("/assignments/{assignment_id}/re-evaluate")
def re_evaluate(assignment_id: str):
    """
    Mark assignment for re-evaluation. Scanner will pick up document.
    """
    logger.info("api.re_evaluate: request for assignment_id=%s", assignment_id)
    coll = db[config.get("mongo", {}).get("prompt_collection", "prompt_builder")]
    doc = coll.find_one({"assignment_id": assignment_id})
    if not doc:
        logger.warning("api.re_evaluate: document not found for assignment_id=%s", assignment_id)
        raise HTTPException(status_code=404, detail="assignment not found")

    run_id = str(uuid.uuid4())
    coll.update_one(
        {"_id": doc["_id"]},
        {"$set": {
            "re_evaluation_requested": {
                "request_id": run_id,
                "requested_by": "api",
                "requested_at": datetime.now(timezone.utc),
                "force": True
            },
            "evaluated": False,
            "evaluation_status": "queued",
            "evaluation_queued_at": datetime.now(timezone.utc)
        }}
    )

    logger.info("api.re_evaluate: queued doc _id=%s run_id=%s", str(doc["_id"]), run_id)
    return {"status": "queued_for_re_evaluation", "job_id": run_id}


@app.post("/assignments/{assignment_id}/evaluate")
def evaluate(assignment_id: str, force: bool = Query(False)):
    """
    Mark an assignment for evaluation.
    """
    logger.info("api.evaluate: request for assignment_id=%s force=%s", assignment_id, force)
    coll = db[config.get("mongo", {}).get("prompt_collection", "prompt_builder")]
    doc = coll.find_one({"assignment_id": assignment_id})
    if not doc:
        logger.warning("api.evaluate: document not found for assignment_id=%s", assignment_id)
        raise HTTPException(status_code=404, detail="assignment not found")

    run_id = str(uuid.uuid4())
    coll.update_one(
        {"_id": doc["_id"]},
        {"$set": {
            "re_evaluation_requested": {
                "request_id": run_id,
                "requested_by": "api",
                "requested_at": datetime.now(timezone.utc),
                "force": bool(force)
            },
            "evaluation_status": "queued",
            "evaluation_queued_at": datetime.now(timezone.utc),
            "evaluated": False
        }}
    )

    logger.info("api.evaluate: queued doc _id=%s run_id=%s", str(doc["_id"]), run_id)
    return {"status": "queued_for_evaluation", "job_id": run_id}


@app.post("/admin/scan-evaluate")
def admin_manual_scan():
    """
    Trigger a single scan pass synchronously (useful for debugging).
    """
    logger.info("api.admin_manual_scan: triggered by admin")
    n = scan_and_process_once()
    logger.info("api.admin_manual_scan: scan processed=%d", n)
    return {"processed": n}


@app.get("/assignments/{assignment_id}/results")
def get_assignment(assignment_id: str):
    """
    Fetch assignment doc and evaluation results.
    """
    coll = db[config.get("mongo", {}).get("prompt_collection", "prompt_builder")]
    doc = coll.find_one({"assignment_id": assignment_id})
    if not doc:
        raise HTTPException(status_code=404, detail="assignment not found")

    # convert ObjectId to string for transport
    doc_copy = dict(doc)
    _id = doc_copy.pop("_id", None)
    if _id is not None:
        doc_copy["_id"] = str(_id)

    # prune raw model text if present and too large (optional)
    if doc_copy.get("model_raw_response") and len(str(doc_copy.get("model_raw_response"))) > 20000:
        doc_copy["model_raw_response_preview"] = str(doc_copy.get("model_raw_response"))[:10000] + " ...[truncated]"
        doc_copy.pop("model_raw_response", None)

    return {
        "assignment_id": assignment_id,
        "evaluated": doc_copy.get("evaluated"),
        "evaluation_status": doc_copy.get("evaluation_status"),
        "evaluation_result": {
            "scores": doc_copy.get("scores"),
            "overall": doc_copy.get("overall"),
            "feedback": doc_copy.get("feedback"),
            "violations": doc_copy.get("violations"),
        },
        "re_evaluation_requested": doc_copy.get("re_evaluation_requested"),
        "raw_doc": doc_copy
    }