# app.py
import json
import re

from fastapi import FastAPI, Query, HTTPException
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
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


def _extract_json_from_llm_raw(raw: str) -> dict:
    """
    Extracts the first valid JSON object from llm_raw,
    even if extra text follows.
    """

    try:
        # Find the first {...} JSON block using a greedy match
        match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
        if not match:
            return {}

        json_str = match.group(0)
        return json.loads(json_str)

    except Exception:
        return {}

def _result_from_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build result object using model_raw_response.llm_raw,
    supporting both OLD and NEW LLM response formats.
    """

    scores = None
    overall = None
    feedback = None
    violations = []

    mrr = doc.get("model_raw_response")
    if mrr:
        llm_raw = mrr.get("llm_raw")
        if llm_raw:
            parsed = _extract_json_from_llm_raw(llm_raw)

            # 🔁 BACKWARD + FORWARD COMPATIBILITY
            # New LLM format
            if "total_score" in parsed:
                overall = parsed.get("total_score")
                scores = parsed.get("grade_breakdown")
                feedback = parsed.get("feedback")
                violations = parsed.get("priority_chunks") or []

            # Old LLM format (fallback)
            else:
                scores = parsed.get("scores")
                overall = parsed.get("overall")
                feedback = parsed.get("feedback")
                violations = parsed.get("violations") or []

    return {
        "assignment_id": doc.get("assignment_id"),
        "student_id": doc.get("student_id"),
        "evaluation_status": doc.get("evaluation_status"),
        "scores": scores,
        "overall": overall,
        "feedback": feedback,
        "violations": violations,
    }

@app.get("/assignments/{job_id}/{assignment_id}/results")
def get_assignment_results(
    job_id: str,
    assignment_id: str,
    student_id: Optional[str] = Query(None),
):
    coll = db[config.get("mongo", {}).get("prompt_collection", "prompt_builder")]

    base_query = {
        "evaluation_job_id": job_id,
        "assignment_id": assignment_id
    }

    if student_id:
        # Single-student mode
        doc = coll.find_one(
            {**base_query, "student_id": student_id},
            sort=[("_id", -1)]
        )
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"assignment {assignment_id} for job {job_id} and student {student_id} not found"
            )
        return _result_from_doc(doc)

    # Batch mode
    cursor = coll.find(base_query)
    results = [_result_from_doc(d) for d in cursor]

    # NEW: compute completed count
    completed_count = sum(1 for r in results if r.get("evaluation_status") == "done")

    return {
        "job_id": job_id,
        "assignment_id": assignment_id,
        "count": len(results),
        **({"completed": completed_count}),
        "results": results,
    }