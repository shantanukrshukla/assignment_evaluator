# module/scanner_db_runner.py
"""
Production-ready scanner/dispatcher that claims candidate docs and invokes worker.process_message.
Keeps claiming semantics lightweight (pre-claim then worker takeover) and emits good logs.
"""
from __future__ import annotations

import time
import threading
from datetime import datetime
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.mongo_client import db
from utils.logging import logger
from utils.config_loader import config
from module.worker import process_message

# Optional MLflow usage (best-effort)
try:
    import mlflow
    _MLFLOW_OK = True
except Exception:
    _MLFLOW_OK = False

# Config-driven defaults
SCAN_INTERVAL_S = int(config.get("scanner", {}).get("interval_s", 30))
BATCH_LIMIT = int(config.get("scanner", {}).get("batch_limit", 10))
WORKER_COUNT = int(config.get("scanner", {}).get("worker_count", 1))

# Use configured prompt collection name (fallback)
COLL_NAME = config.get("mongo", {}).get("prompt_collection") or "prompt_builder"


def _candidate_query() -> Dict[str, Any]:
    """
    Documents that either haven't been evaluated, are explicitly requested for re-eval,
    or are in a recoverable state (queued/error) are candidates.
    """
    return {
        "$and": [
            {
                "$or": [
                    {"evaluated": {"$exists": False}},
                    {"evaluated": False},
                    {"re_evaluation_requested.request_id": {"$exists": True}}
                ]
            },
            {
                "$or": [
                    {"evaluation_status": {"$exists": False}},
                    {"evaluation_status": {"$in": ["queued", "error", None]}},
                    {"evaluation_status": {"$ne": "in_progress"}}
                ]
            }
        ]
    }


def _atomic_preclaim(coll, oid) -> bool:
    """
    Light pre-claim to mark document as queued and increment attempts.
    Returns True when the update modified a document (candidate accepted).
    """
    try:
        res = coll.update_one(
            {"_id": oid,
             "$or": [
                 {"evaluation_status": {"$exists": False}},
                 {"evaluation_status": {"$in": ["queued", "error", None]}},
                 {"evaluation_status": {"$ne": "in_progress"}}
             ]},
            {"$set": {"evaluation_status": "queued", "evaluation_queued_at": datetime.utcnow()},
             "$inc": {"evaluation_attempts": 1}}
        )
        claimed = getattr(res, "modified_count", 0) > 0 or getattr(res, "matched_count", 0) > 0
        logger.debug("scanner._atomic_preclaim: _id=%s claimed=%s matched=%s modified=%s",
                     str(oid), claimed, getattr(res, "matched_count", -1), getattr(res, "modified_count", -1))
        return claimed
    except Exception:
        logger.exception("scanner._atomic_preclaim failed for %s", str(oid))
        return False


def _release_claim_mark_error(coll, oid, reason: str = "scanner_error") -> None:
    try:
        coll.update_one({"_id": oid},
                        {"$set": {"evaluation_status": "error", "evaluation_error": reason, "evaluation_error_ts": datetime.utcnow()}})
        logger.info("scanner._release_claim_mark_error: marked _id=%s error=%s", str(oid), reason)
    except Exception:
        logger.exception("scanner._release_claim_mark_error: failed for %s", str(oid))


def _make_message_for_doc(doc) -> Dict[str, Any]:
    """
    Construct a message for the worker.
    Important: include skip_claim=True to signal the worker that the scanner already pre-claimed
    the document (so the worker should not re-run the filtered atomic claim which can race).
    """
    return {
        "doc_id": str(doc["_id"]),
        "assignment_id": doc.get("assignment_id"),
        "force": bool(doc.get("re_evaluation_requested", {}).get("force", False)),
        "requested_by": "scanner",
        "skip_claim": True
    }


def process_one_doc(doc: Dict[str, Any]) -> bool:
    """
    Claim doc (light pre-claim), then call process_message synchronously.
    Returns True when processing succeeded (worker didn't raise).
    """
    coll = db[COLL_NAME]
    oid = doc["_id"]
    assignment_id = doc.get("assignment_id")
    logger.info("scanner.process_one_doc: candidate _id=%s assignment=%s", str(oid), assignment_id)
    start = time.time()

    # light pre-claim to mark queued
    claimed = _atomic_preclaim(coll, oid)
    if not claimed:
        logger.info("scanner.process_one_doc: skip _id=%s (pre-claim failed)", str(oid))
        return False

    try:
        msg = _make_message_for_doc(doc)
        logger.info("scanner.process_one_doc: invoking process_message for _id=%s assignment=%s", str(oid), assignment_id)
        t0 = time.time()
        process_message(msg)
        duration = time.time() - t0
        logger.info("scanner.process_one_doc: finished process_message for _id=%s assignment=%s duration=%.3fs", str(oid), assignment_id, duration)
        return True
    except Exception:
        logger.exception("scanner.process_one_doc: processing failed for _id=%s", str(oid))
        _release_claim_mark_error(coll, oid, reason="scanner_processing_exception")
        return False
    finally:
        logger.debug("scanner.process_one_doc: total time for _id=%s = %.3fs", str(oid), time.time() - start)


def _fetch_candidates(limit: int) -> List[Dict[str, Any]]:
    coll = db[COLL_NAME]
    q = _candidate_query()
    logger.debug("scanner._fetch_candidates: running query=%s limit=%d", q, limit)
    cursor = coll.find(q).sort([("evaluation_queued_at", 1), ("created_at", 1)]).limit(limit)
    logger.info(f"records : {cursor}")
    docs = list(cursor)
    logger.info("scanner._fetch_candidates: docs=%s", docs)
    logger.info("scanner._fetch_candidates: found %d candidate docs", len(docs))
    return docs


def scan_and_process_once(limit: int = BATCH_LIMIT, concurrency: int = WORKER_COUNT) -> int:
    docs = _fetch_candidates(limit=limit)
    if not docs:
        return 0

    ids = [str(d.get("_id")) for d in docs]
    logger.debug("scanner.scan_and_process_once: candidate_ids=%s", ids)

    processed = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futures = {ex.submit(process_one_doc, d): d for d in docs}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                ok = fut.result()
                if ok:
                    processed += 1
            except Exception:
                logger.exception("scanner.scan_and_process_once: unexpected future exception for %s", str(d.get("_id")))
    logger.info("scanner.scan_and_process_once: processed %d of %d candidates", processed, len(docs))

    # Optional MLflow metric
    if _MLFLOW_OK:
        try:
            mlflow.log_metric("scanner_last_processed", processed)
        except Exception:
            logger.exception("scanner.scan_and_process_once: mlflow logging failed (continuing)")

    return processed


def scanner_loop(stop_event: threading.Event):
    logger.info("scanner.scanner_loop: starting loop; interval=%ds batch=%d concurrency=%d collection=%s",
                SCAN_INTERVAL_S, BATCH_LIMIT, WORKER_COUNT, COLL_NAME)
    while not stop_event.is_set():
        try:
            n = scan_and_process_once(limit=BATCH_LIMIT, concurrency=WORKER_COUNT)
            if n:
                logger.info("scanner.scanner_loop: processed %d docs this pass", n)
            else:
                logger.debug("scanner.scanner_loop: no docs processed this pass")
        except Exception:
            logger.exception("scanner.scanner_loop: error during scan pass")
        stop_event.wait(SCAN_INTERVAL_S)
    logger.info("scanner.scanner_loop: stopping loop")
