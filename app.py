# app.py
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Query, HTTPException

from utils.config_loader import config
from utils.mongo_client import db
from utils.logging import logger
from module.scanner_db_runner import scanner_loop, scan_and_process_once

app = FastAPI(title="Assignment Evaluator")

# ---------------------------------------------------------------------
# Scanner thread globals
# ---------------------------------------------------------------------
_SCANNER_THREAD: Optional[threading.Thread] = None
_SCANNER_STOP_EVENT: Optional[threading.Event] = None


# ---------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    logger.info("Starting Assignment Evaluator")
    if config.get("scanner", {}).get("start_in_app", True):
        global _SCANNER_THREAD, _SCANNER_STOP_EVENT
        _SCANNER_STOP_EVENT = threading.Event()
        _SCANNER_THREAD = threading.Thread(
            target=scanner_loop,
            args=(_SCANNER_STOP_EVENT,),
            daemon=True
        )
        _SCANNER_THREAD.start()


@app.on_event("shutdown")
def shutdown_event():
    global _SCANNER_THREAD, _SCANNER_STOP_EVENT
    if _SCANNER_STOP_EVENT:
        _SCANNER_STOP_EVENT.set()
    if _SCANNER_THREAD:
        _SCANNER_THREAD.join(timeout=5)


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    try:
        db.list_collection_names()
        return {"status": "ok", "mongo": True}
    except Exception:
        return {"status": "degraded", "mongo": False}


# ---------------------------------------------------------------------
# Evaluation APIs
# ---------------------------------------------------------------------
@app.post("/assignments/{assignment_id}/re-evaluate")
def re_evaluate(assignment_id: str):
    coll = db[config.get("mongo", {}).get("prompt_collection", "prompt_builder")]
    doc = coll.find_one({"assignment_id": assignment_id})
    if not doc:
        raise HTTPException(404, "assignment not found")

    run_id = str(uuid.uuid4())
    coll.update_one(
        {"_id": doc["_id"]},
        {"$set": {
            "evaluation_status": "queued",
            "evaluated": False,
            "evaluation_queued_at": datetime.now(timezone.utc)
        }}
    )
    return {"status": "queued", "job_id": run_id}


@app.post("/assignments/{assignment_id}/evaluate")
def evaluate(assignment_id: str):
    return re_evaluate(assignment_id)


@app.post("/admin/scan-evaluate")
def admin_manual_scan():
    return {"processed": scan_and_process_once()}


# ---------------------------------------------------------------------
# LLM FIELD EXTRACTORS (PRODUCTION-SAFE)
# ---------------------------------------------------------------------
def _extract_total_score(raw: str) -> Optional[float]:
    m = re.search(r'"total_score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw)
    return float(m.group(1)) if m else None

def _extract_scores(raw: str) -> Optional[Dict[str, float]]:
    """
    Extract rubric scores from LLM output.
    Supports multiple schema versions:
      - parameter_scores (new)
      - score_breakdown (old)
    """
    if not raw:
        return None

    # Prefer NEW key
    for key in ("parameter_scores", "score_breakdown","grade_breakdown"):
        m = re.search(
            rf'"{key}"\s*:\s*\{{(.*?)\}}',
            raw,
            re.DOTALL
        )
        if not m:
            continue

        block = m.group(1)
        scores: Dict[str, float] = {}

        for name, val in re.findall(
            r'"([^"]+)"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            block
        ):
            scores[name] = float(val)

        if scores:
            return scores

    return None



def _extract_feedback(raw: str) -> Optional[str]:
    m = re.search(r'"feedback"\s*:\s*\{(.*?)\}\s*,', raw, re.DOTALL)
    if not m:
        return None

    block = m.group(1)
    parts: List[str] = []

    for section, text in re.findall(
        r'"([^"]+)"\s*:\s*"([^"]*)"',
        block
    ):
        parts.append(f"### {section}")
        parts.append(f"- {text}")
        parts.append("")

    return "\n".join(parts).strip() if parts else None


def _extract_passed_status(raw: str) -> Optional[str]:
    total = _extract_total_score(raw)
    if total is None:
        return None

    m = re.search(r'"passing_score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw)
    if not m:
        return None

    passing = float(m.group(1))
    return "Pass" if total >= passing else "Fail"


# ---------------------------------------------------------------------
# Result builder (UI CONTRACT)
# ---------------------------------------------------------------------
def _result_from_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    UI-facing result.
    Schema-agnostic, production-safe.
    """

    llm_raw = doc.get("model_raw_response", {}).get("llm_raw") or ""

    scores = _extract_scores(llm_raw)
    overall = _extract_total_score(llm_raw)
    feedback = _extract_feedback(llm_raw)
    passed_status = (
        _extract_passed_status(llm_raw)
        or doc.get("passed_status")
    )

    return {
        "assignment_id": doc.get("assignment_id"),
        "student_id": doc.get("student_id"),
        "evaluation_status": doc.get("evaluation_status"),
        "scores": scores,
        "overall": overall,
        "feedback": feedback,
        "violations": [],
        "passed_status": passed_status,
    }


# ---------------------------------------------------------------------
# Results API
# ---------------------------------------------------------------------
@app.get("/assignments/{job_id}/{assignment_id}/results")
def get_assignment_results(
    job_id: str,
    assignment_id: str,
    student_id: Optional[str] = Query(None),
):
    coll = db[config.get("mongo", {}).get("prompt_collection", "prompt_builder")]

    query = {
        "evaluation_job_id": job_id,
        "assignment_id": assignment_id,
    }
    if student_id:
        query["student_id"] = student_id

    docs = list(coll.find(query).sort("_id", -1))

    if student_id:
        if not docs:
            raise HTTPException(404, "result not found")
        return _result_from_doc(docs[0])

    results = [_result_from_doc(d) for d in docs]
    return {
        "job_id": job_id,
        "assignment_id": assignment_id,
        "count": len(results),
        "completed": sum(1 for r in results if r["evaluation_status"] == "done"),
        "results": results,
    }
