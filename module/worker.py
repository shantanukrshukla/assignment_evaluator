# module/worker.py (FULL updated - prompt_builder driven evaluation)
from __future__ import annotations
import json
import re
import ast
import hashlib
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from bson import ObjectId

from utils.mongo_client import db
from utils.logging import logger
from utils.config_loader import config
from parser.summarizer import extractive_summary, top_keywords
from llm_model.llm_connector import ask_llm
from utils.parser import _fast_html_to_text, normalize_course_payload
from utils.mlflow_init import init_mlflow

# Optional LangChain PromptTemplate
try:
    from langchain import PromptTemplate
except Exception:
    PromptTemplate = None

# Optional MLflow
try:
    import mlflow
    _MLFLOW_OK = True
except Exception:
    mlflow = None
    _MLFLOW_OK = False

# Worker identity
WORKER_ID = f"worker-{int(time.time())}"

# ----------------------------
# Config-driven thresholds & defaults
# ----------------------------
_TOKEN_BUDGET = int(config.get("retrieval", {}).get("token_budget", 350000))
SUMMARY_THRESHOLD = int(_TOKEN_BUDGET * 0.5)

MAX_ATTEMPTS = int(config.get("job_manager", {}).get("task_retries", 1)) + 3
RELEVANCE_THRESHOLD = float(config.get("retrieval", {}).get("min_similarity", 0.05))

COLL_NAME = config.get("mongo", {}).get("prompt_collection", "prompt_builder")

DEBUG_PROMPTS = bool(config.get("debug", {}).get("debug_prompts", False))
PERSIST_PROMPTS_TO_DB = bool(config.get("debug", {}).get("persist_prompts_in_db", False))
PROMPT_TRUNCATE = int(config.get("debug", {}).get("prompt_truncate_chars", 500000))

_LLM_CFG = config.get("llm", {}) or {}
SUBMISSION_MAX_CHARS = int(_LLM_CFG.get("submission_max_chars", 800000))
SYSTEM_PROMPT_MAX_CHARS = int(_LLM_CFG.get("system_prompt_max_chars", 6000000))
DEFAULT_MAX_CHARS = int(_LLM_CFG.get("default_max_chars", 4000000))
COMMENT_MAX_CHARS = int(_LLM_CFG.get("max_comment_chars", 2000000))

# ----------------------------
# Utility helpers (parsing, cleaning, normalization)
# ----------------------------

def _compute_dedupe_key(text: str, prompt_version: str) -> str:
    h = hashlib.sha256()
    h.update((prompt_version or "").encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _clean_and_truncate_text(raw_text: Optional[str], max_chars: int = DEFAULT_MAX_CHARS) -> str:
    if not raw_text:
        return ""
    try:
        cleaned = _fast_html_to_text(raw_text)
        cleaned = " ".join(cleaned.split())
        if len(cleaned) > max_chars:
            return cleaned[:max_chars] + " ...[truncated]"
        return cleaned
    except Exception:
        try:
            s = str(raw_text)
            return (s[:max_chars] + " ...[truncated]") if len(s) > max_chars else s
        except Exception:
            return ""


def _safe_parse_json(text: Optional[str]) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        matches = re.findall(r"\{[\s\S]*?\}", text)
        if matches:
            cand = max(matches, key=len)
            return json.loads(cand)
    except Exception:
        pass
    try:
        t = text.strip()
        t = re.sub(r"```(?:json)?\n?|```\n?", "", t, flags=re.IGNORECASE)
        t = t.replace("\u201c", '"').replace('\u201d', '"')
        t = re.sub(r"(?P<prefix>[:\s,{[]?)'([^']*?)'(?P<suffix>[,\]\}\s])", r'\1"\2"\3', t)
        t = re.sub(r",(\s*[\]\}])", r"\1", t)
        return json.loads(t)
    except Exception:
        pass
    try:
        t2 = re.sub(r"```. *?```", "", text, flags=re.DOTALL)
        obj = ast.literal_eval(t2)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass
    return None


def _is_number_like(v):
    try:
        float(v)
        return True
    except Exception:
        return False


def _normalize_numeric_map(num_map: Dict[str, Any]) -> Dict[str, float]:
    nums = {}
    for k, v in (num_map or {}).items():
        if _is_number_like(v):
            try:
                nums[k] = float(v)
            except Exception:
                pass
    if not nums:
        return {}
    max_v = max(abs(x) for x in nums.values() if x is not None)
    if max_v == 0:
        return {k: 0.0 for k in nums}
    if max_v <= 1.0:
        return {k: round(max(0.0, min(1.0, v)), 4) for k, v in nums.items()}
    if max_v <= 10.0:
        scale = 10.0
    elif max_v <= 100.0:
        scale = 100.0
    else:
        scale = max_v
    return {k: round(max(0.0, min(1.0, v / scale)), 4) for k, v in nums.items()}


def _validate_and_normalize_final(final_obj: Dict[str, Any], rubric_logic: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(final_obj, dict):
        return None

    # Prefer point-based grading if present
    total_score = final_obj.get("total_score")
    grade_breakdown = final_obj.get("grade_breakdown")
    if total_score is not None and isinstance(grade_breakdown, dict):
        # Use points directly
        scores = {k: float(v) for k, v in grade_breakdown.items() if _is_number_like(v)}
        overall = float(total_score)
        # Feedback extraction (robust)
        feedback = final_obj.get("feedback")
        if isinstance(feedback, dict):
            strengths = feedback.get("strengths")
            areas = feedback.get("areas_for_improvement")
            feedback_str = ""
            if strengths:
                feedback_str += "Strengths: " + "; ".join(strengths) + ". "
            if areas:
                feedback_str += "Areas for improvement: " + "; ".join(areas) + "."
            feedback = feedback_str.strip()
        elif isinstance(feedback, list):
            feedback = "; ".join(str(f) for f in feedback)
        elif isinstance(feedback, str):
            feedback = feedback
        else:
            feedback = ""
        # Violations extraction
        violations = final_obj.get("violations")
        if not violations:
            violations = final_obj.get("areas_for_improvement") or []
        if isinstance(violations, str):
            violations = [violations]
        if violations is None:
            violations = []
        return {
            "scores": scores,
            "overall": overall,
            "feedback": feedback,
            "violations": violations
        }

    raw_scores = final_obj.get("scores") or {}
    if not isinstance(raw_scores, dict):
        candidate = {}
        for k, v in final_obj.items():
            if k in ("overall", "feedback", "violations"):
                continue
            if _is_number_like(v):
                candidate[str(k)] = float(v)
        raw_scores = candidate

    normalized_scores = _normalize_numeric_map(raw_scores) if raw_scores else {}

    overall = final_obj.get("overall")
    normalized_overall = None
    if overall is not None and _is_number_like(overall):
        try:
            ov = float(overall)
            if 0.0 <= ov <= 1.0:
                normalized_overall = round(ov, 4)
            else:
                if abs(ov) <= 10:
                    normalized_overall = round(max(0.0, min(1.0, ov / 10.0)), 4)
                elif abs(ov) <= 100:
                    normalized_overall = round(max(0.0, min(1.0, ov / 100.0)), 4)
                else:
                    normalized_overall = round(max(0.0, min(1.0, ov / abs(ov))), 4) if ov != 0 else 0.0
        except Exception:
            normalized_overall = None

    feedback = final_obj.get("feedback") or ""
    violations = final_obj.get("violations") or []
    if isinstance(violations, str):
        violations = [violations]
    if violations is None:
        violations = []

    if not normalized_scores and normalized_overall is None:
        return None

    return {
        "scores": normalized_scores if normalized_scores else None,
        "overall": normalized_overall,
        "feedback": feedback,
        "violations": violations
    }


def _append_memory_for_doc(coll, oid, scores_obj, feedback, chunk_results):
    try:
        coll.update_one(
            {"_id": oid},
            {
                "$push": {
                    "memory.past_scores": {"$each": [scores_obj], "$slice": -50},
                    "memory.past_feedback": {"$each": [feedback], "$slice": -50},
                    "memory.past_chunk_results": {
                        "$each": [{"ts": datetime.now(timezone.utc).isoformat(), "chunks": chunk_results}],
                        "$slice": -20}
                },
                "$set": {"memory.last_evaluated_at": datetime.now(timezone.utc)}
            }
        )
    except Exception:
        logger.exception("_append_memory_for_doc: failed to update memory for %s", str(oid))


# ----------------------------
# Core evaluation flow (prompt_builder-driven)
# ----------------------------

def evaluate_document(doc: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prompt-builder driven evaluation.
    Steps:
      - read system_prompt and user_prompt from prompt_builder document
      - use LangChain PromptTemplate if present to assemble a stronger system prompt
      - call ask_llm(system_prompt, user_prompt)
      - parse & normalize
      - persist results back into prompt_builder collection

    The function intentionally avoids inserting course chunks into prompts.
    """
    start_all = time.time()
    coll = db[COLL_NAME]
    doc_id = str(doc.get("_id"))
    assignment_id = doc.get("assignment_id")
    logger.info("evaluate_document (pb-driven): start doc=%s assignment=%s", doc_id, assignment_id)

    prompt_version = doc.get("prompt_version") or "v1"

    # Extract submission text
    submission = doc.get("extracted_excerpt") or doc.get("extractedExcerpt") or doc.get("submission") or ""
    if submission:
        logger.info("evaluate_document: found submission text (len=%d) and doc=%s", len(submission), doc)
    else:
        logger.warning("evaluate_document: submission text is EMPTY for doc=%s and doc =%s", doc_id, doc)

    submission_trim = _clean_and_truncate_text(submission, max_chars=SUBMISSION_MAX_CHARS)

    # Pull authoritative prompts from prompt_builder document
    doc_system_prompt = (doc.get("system_prompt") or doc.get("systemPrompt") or "").strip()
    doc_user_prompt = (doc.get("user_prompt") or doc.get("userPrompt") or "").strip()
    rubric_logic = _stringify_rubric_logic(doc.get("rubric_logic"))

    if not doc_system_prompt and not doc_user_prompt:
        logger.warning("evaluate_document: prompt_builder doc missing both system_prompt and user_prompt for %s", doc_id)

    # Build stronger system prompt using LangChain PromptTemplate when available
    creative_system_template_text = (
        "You are an imaginative, strict, and highly-accurate automated assignment evaluator.\n"
        "Follow the rubric precisely when present. Be creative in phrasing feedback but rigorous in scoring.\n"
        "-- Authoritative rubric (if provided) --\n{rubric}\n\n"
        "Rules:\n1) Use ONLY the information in the student submission and the rubric provided above.\n"
        "2) Do NOT reference external sources, textbooks, or the internet.\n"
        "3) Provide concise, actionable feedback that an instructor or student can act on.\n"
        "4) Optionally include a top-level key 'priority_chunks' with an empty array (we are not providing chunks currently).\n"
    )

    if PromptTemplate is not None:
        try:
            tpl = PromptTemplate(input_variables=["rubric"], template=creative_system_template_text)
            system_prompt_final = tpl.format(rubric=rubric_logic or "(none provided)")
        except Exception:
            logger.exception("evaluate_document: LangChain PromptTemplate formatting failed; using fallback formatting")
            system_prompt_final = creative_system_template_text.replace('{rubric}', rubric_logic or "(none provided)").replace('{doc_system}', doc_system_prompt or "(none provided)")
    else:
        system_prompt_final = creative_system_template_text.replace('{rubric}', rubric_logic or "(none provided)").replace('{doc_system}', doc_system_prompt or "(none provided)")

    # Truncate system prompt to configured max length
    if len(system_prompt_final) > SYSTEM_PROMPT_MAX_CHARS:
        system_prompt_final = system_prompt_final + " ...[system_prompt_truncated]"

    # Build user prompt (prefer doc_user_prompt if provided)
    if doc_user_prompt:
        user_prompt_final = doc_user_prompt + "\n\nStudent submission:\n" + submission_trim
    else:
        user_prompt_final = "Student submission:\n" + submission_trim

    # Persist the prompts used for traceability (best-effort)
    try:
        coll.update_one({"_id": doc.get("_id")}, {"$set": {
            "last_used_prompts.system_prompt": system_prompt_final[:10000],
            "last_used_prompts.user_prompt": user_prompt_final[:10000],
            "last_used_prompts.ts": datetime.now(timezone.utc).isoformat()
        }})
    except Exception:
        logger.exception("evaluate_document: failed to persist last_used_prompts for doc=%s", doc_id)

    # Call LLM for evaluation
    provider_override = (message.get("llm_provider") or doc.get("llm_provider") or None)
    try:
        llm_raw = ask_llm(system_prompt=system_prompt_final, user_prompt=user_prompt_final, provider_override=provider_override)
    except Exception:
        logger.exception("evaluate_document: ask_llm failed for doc=%s", doc_id)
        llm_raw = None

    if DEBUG_PROMPTS:
        logger.debug("DEBUG_EVAL_RAW (trunc %d): %s", PROMPT_TRUNCATE, (llm_raw or "")[:PROMPT_TRUNCATE])

    # parse screening output
    parsed = _safe_parse_json(llm_raw) or {}
    # some providers may wrap result under 'final'
    final_obj = parsed.get('final') if isinstance(parsed, dict) and 'final' in parsed else parsed

    normalized_final = _validate_and_normalize_final(final_obj, rubric_logic=rubric_logic)

    # If normalization failed, attempt a reformat pass to coerce JSON-only output
    if normalized_final is None:
        logger.warning("evaluate_document: initial parse/normalize failed for doc=%s; attempting reformat", doc_id)
        reformat_system = (
            "You are a strict JSON formatter. Output exactly one JSON object and nothing else."
        )
        reformat_user = (
            "Extract numeric scores, overall, feedback and violations from the original assistant output below and return exactly one JSON object following schema:\n"
            "{\"scores\": {\"<criterion_id>\": 1 -10, ...} | null, \"overall\": 1-10 | null, \"feedback\": \"string\", \"violations\": [\"string\", ...]}\n\nOriginal output:\n" + (llm_raw or "")
        )
        try:
            reformat_raw = ask_llm(system_prompt=reformat_system, user_prompt=reformat_user, provider_override=provider_override)
            reformat_json = _safe_parse_json(reformat_raw) or {}
            reformat_final = reformat_json.get("final") if isinstance(reformat_json, dict) and "final" in reformat_json else reformat_json
            normalized_final = _validate_and_normalize_final(reformat_final, rubric_logic=rubric_logic)
            if normalized_final:
                logger.info("evaluate_document: reformat pass succeeded for doc=%s", doc_id)
        except Exception:
            logger.exception("evaluate_document: reformat attempt failed for doc=%s", doc_id)

    # Fallback if still not normalized
    if normalized_final is None:
        logger.warning("evaluate_document: normalized_final STILL None for doc=%s; producing parse_failure result", doc_id)
        normalized_final = {"scores": None, "overall": None, "feedback": "Unable to parse LLM output into expected schema.", "violations": ["parse_failure"]}

    # Build chunk_results placeholder (we are not performing per-chunk eval in this version)
    chunk_results = []

    # Build result object
    result = {
        "scores": normalized_final.get("scores"),
        "overall": normalized_final.get("overall"),
        "feedback": normalized_final.get("feedback"),
        "violations": normalized_final.get("violations"),
        "chunk_results": chunk_results,
        "screening_raw": llm_raw,
        "screening_json": parsed,
        "dedupe_key": _compute_dedupe_key((doc_system_prompt or "") + (doc_user_prompt or "") + submission_trim, prompt_version),
        "prompt_version": prompt_version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model_raw_response": {"llm_raw": llm_raw}
    }

    logger.info("evaluate_document: finished doc=%s assignment=%s elapsed=%.3fs", doc_id, assignment_id, time.time() - start_all)

    return result


# ----------------------------
# Process message (entrypoint) - claims, runs evaluate_document and persists results
# ----------------------------

def process_message(message: Dict[str, Any]):
    from pymongo import ReturnDocument
    coll = db[COLL_NAME]

    jobs_coll_name = config.get("mongo", {}).get("jobs_collection", "jobs")
    jobs_coll = db[jobs_coll_name]

    doc_id_raw = message.get("doc_id") or message.get("assignment_id")
    force = bool(message.get("force", False))
    requester = message.get("requested_by", "pipe")
    start_proc = time.time()

    logger.info("process_message: entry doc_id_raw=%s requester=%s force=%s", str(doc_id_raw), requester, force)

    if not doc_id_raw:
        logger.warning("process_message: missing doc_id in message: %s", message)
        return

    # Lookup doc by _id or assignment_id
    doc = None
    try:
        oid = ObjectId(doc_id_raw)
        doc = coll.find_one({"_id": oid})
        logger.debug("process_message: looked up by _id=%s", str(oid))
    except Exception:
        doc = coll.find_one({"assignment_id": doc_id_raw})
        logger.debug("process_message: looked up by assignment_id=%s", doc_id_raw)

    if not doc:
        logger.warning("process_message: document not found for id %s", doc_id_raw)
        return

    logger.info(
        "process_message: retrieved doc _id=%s assignment_id=%s evaluation_status=%s evaluated=%s attempts=%s",
        str(doc.get("_id")), doc.get("assignment_id"), doc.get("evaluation_status"), doc.get("evaluated"),
        doc.get("evaluation_attempts")
    )

    # dedupe check
    prompt_version = doc.get("prompt_version") or "v1"
    # Use extracted_excerpt (which contains student submission from file or description field)
    submission_text = doc.get("extracted_excerpt") or doc.get("extractedExcerpt") or doc.get("submission") or ""
    source_text = (doc.get("system_prompt") or "") + (doc.get("user_prompt") or "") + submission_text
    dedupe_key = _compute_dedupe_key(source_text, prompt_version)
    existing_dk = doc.get("dedupe_key")
    logger.debug("process_message: computed dedupe_key=%s existing=%s", dedupe_key[:12], (existing_dk or "")[:12])

    if existing_dk == dedupe_key and doc.get("evaluated") and not force:
        logger.info("process_message: skipping evaluation for %s (dedupe match)", doc_id_raw)
        coll.update_one({"_id": doc["_id"]}, {"$push": {
            "evaluation_job_history": {"run_id": str(time.time()), "status": "skipped", "dedupe_key": dedupe_key, "timestamp": datetime.now(timezone.utc)}
        }})
        return

    # Attempt atomic claim (similar to original semantics)
    claim_attempts = int(config.get("job_manager", {}).get("claim_retries", 3))
    claim_delay = float(config.get("job_manager", {}).get("claim_backoff_seconds", 0.5))
    claimed_doc = None

    if message.get("skip_claim"):
        try:
            res = coll.find_one_and_update(
                {"_id": doc["_id"]},
                {"$set": {"evaluation_status": "in_progress", "evaluation_started_at": datetime.now(timezone.utc), "evaluation_worker_id": WORKER_ID}, "$inc": {"evaluation_attempts": 1}},
                return_document=ReturnDocument.AFTER,
            )
            if res:
                claimed_doc = res
                logger.debug("process_message: takeover claim applied for %s (skip_claim)", doc_id_raw)
            else:
                logger.warning("process_message: takeover claim failed for %s (skip_claim)", doc_id_raw)
        except Exception:
            logger.exception("process_message: takeover claim exception for %s (skip_claim)", doc_id_raw)
    else:
        claim_filter = {"_id": doc["_id"], "$or": [{"evaluation_status": {"$exists": False}}, {"evaluation_status": "queued"}, {"evaluation_status": "error"}]}
        claim_update = {"$set": {"evaluation_status": "in_progress", "evaluation_started_at": datetime.now(timezone.utc), "evaluation_worker_id": WORKER_ID}, "$inc": {"evaluation_attempts": 1}}

        for attempt in range(1, max(1, claim_attempts) + 1):
            try:
                res = coll.find_one_and_update(claim_filter, claim_update, return_document=ReturnDocument.AFTER)
                if res:
                    claimed_doc = res
                    logger.info("process_message: atomic claim succeeded for doc=%s on attempt=%d", doc_id_raw, attempt)
                    break
                else:
                    logger.info("process_message: atomic claim attempt=%d failed for doc=%s (likely claimed by another worker)", attempt, doc_id_raw)
            except Exception:
                logger.exception("process_message: atomic claim attempt=%d raised exception for doc=%s", attempt, doc_id_raw)
            if attempt < claim_attempts:
                try:
                    time.sleep(claim_delay * attempt)
                except Exception:
                    pass

    if not claimed_doc:
        logger.info("process_message: skip %s (pre-claim failed after %d attempts)", str(doc_id_raw), claim_attempts)
        coll.update_one({"_id": doc["_id"]}, {"$push": {"evaluation_job_history": {"run_id": str(time.time()), "status": "pre_claim_failed", "attempts": claim_attempts, "timestamp": datetime.now(timezone.utc)}}})
        return

    run_id = str(int(time.time() * 1000))
    run_entry = {"run_id": run_id, "requested_by": requester, "requested_at": datetime.now(timezone.utc), "status": "in_progress", "dedupe_key": dedupe_key, "prompt_version": prompt_version}
    logger.info("process_message: claimed doc _id=%s run_id=%s worker_id=%s", str(doc["_id"]), run_id, WORKER_ID)

    # Update jobs collection -> mark job as in_progress (best-effort)
    try:
        job_filter = {}
        job_id = message.get("evaluation_job_id") or doc.get("evaluation_job_id")
        if job_id:
            job_filter = {"evaluation_job_id": job_id}
        else:
            job_filter = {"assignment_id": str(doc.get("assignment_id")), "status": "prompt_built"}

        job_update = {"$set": {"status": "in_progress", "started_by": WORKER_ID, "started_at": datetime.now(timezone.utc)}, "$push": {"history": {"event": "claimed", "run_id": run_id, "worker": WORKER_ID, "ts": datetime.now(timezone.utc)}}}
        job_res = jobs_coll.find_one_and_update(job_filter, job_update, return_document=ReturnDocument.AFTER)
        if job_res:
            logger.info("process_message: jobs collection updated to in_progress for job=%s", job_res.get("evaluation_job_id"))
        else:
            logger.warning("process_message: no matching job found to mark in_progress with filter=%s", job_filter)
    except Exception:
        logger.exception("process_message: failed to update jobs collection to in_progress (continuing)")

    # Initialize MLflow (best-effort)
    if _MLFLOW_OK:
        try:
            init_mlflow()
        except Exception:
            logger.exception("process_message: init_mlflow failed (continuing)")

    # Only start a new MLflow run if there is NOT already an active run.
    # If an active run exists, reuse it (do not start or end it here).
    mlflow_run_started = False
    if _MLFLOW_OK:
        try:
            active_run = None
            try:
                active_run = mlflow.active_run()
            except Exception:
                active_run = None

            if active_run is None:
                try:
                    mlflow.start_run(run_name=run_id)
                    mlflow_run_started = True
                    logger.debug("process_message: mlflow.start_run created a new run for run_id=%s", run_id)
                except Exception:
                    logger.exception("process_message: mlflow.start_run failed (continuing)")
                    mlflow_run_started = False
            else:
                logger.debug("process_message: detected existing active mlflow run (id=%s); reusing it", getattr(active_run, 'info', None))
                mlflow_run_started = False

            # Safe to call tagging/logging whether we started or reused the run.
            try:
                mlflow.set_tag("run_id", run_id)
                mlflow.set_tag("worker_id", WORKER_ID)
                mlflow.log_param("assignment_id", doc.get("assignment_id") or "unknown")
            except Exception:
                logger.exception("process_message: mlflow logging failed (continuing)")
        except Exception:
            logger.exception("process_message: mlflow active_run check/start failed (continuing)")
            mlflow_run_started = False

    try:
        eval_start = time.time()
        eval_result = evaluate_document(doc, message)
        eval_duration = time.time() - eval_start
        logger.info("process_message: evaluation complete for doc=%s run_id=%s duration=%.3fs", str(doc.get("_id") or doc.get("assignment_id") or "<unknown>"), run_id, eval_duration)

        run_entry.update({
            "status": "done",
            "started_at": datetime.now(timezone.utc),
            "completed_at": datetime.now(timezone.utc),
            "scores": eval_result.get("scores"),
            "overall": eval_result.get("overall"),
            "feedback": eval_result.get("feedback"),
            "violations": eval_result.get("violations"),
            "chunk_results": eval_result.get("chunk_results")
        })

        model_raw = eval_result.get("model_raw_response") or {"screening_raw": eval_result.get("screening_raw"), "chunk_results": eval_result.get("chunk_results")}

        ok = coll.update_one(
            {"_id": doc["_id"], "evaluation_status": "in_progress", "evaluation_worker_id": WORKER_ID},
            {"$push": {"evaluation_job_history": run_entry},
             "$set": {"scores": eval_result.get("scores"),
                      "overall": eval_result.get("overall"),
                      "feedback": eval_result.get("feedback"),
                      "violations": eval_result.get("violations"),
                      "evaluated": True,
                      "evaluation_status": "done",
                      "evaluation_completed_at": datetime.now(timezone.utc),
                      "model_raw_response": model_raw,
                      "dedupe_key": eval_result.get("dedupe_key") or doc.get("dedupe_key")}}
        )
        if getattr(ok, "modified_count", 0) == 0:
            logger.warning("process_message: final update race for doc=%s; appending history only", str(doc["_id"]))
            coll.update_one({"_id": doc["_id"]}, {"$push": {"evaluation_job_history": run_entry}})
        else:
            logger.info("process_message: final update succeeded for doc=%s", str(doc["_id"]))

        _append_memory_for_doc(coll, doc["_id"], eval_result.get("scores"), eval_result.get("feedback"), eval_result.get("chunk_results"))

        # Update jobs collection -> mark job as completed (best-effort)
        try:
            job_filter = {}
            job_id = message.get("evaluation_job_id") or doc.get("evaluation_job_id")
            if job_id:
                job_filter = {"evaluation_job_id": job_id}
            else:
                job_filter = {"assignment_id": str(doc.get("assignment_id")), "status": "in_progress"}

            job_update = {"$set": {"status": "completed", "completed_at": datetime.now(timezone.utc), "duration_seconds": round(eval_duration, 3), "result_summary": {"overall": eval_result.get("overall"), "scores": eval_result.get("scores")}}, "$push": {"history": {"event": "completed", "run_id": run_id, "worker": WORKER_ID, "ts": datetime.now(timezone.utc)}}}
            job_res = jobs_coll.find_one_and_update(job_filter, job_update, return_document=ReturnDocument.AFTER)
            if job_res:
                logger.info("process_message: jobs collection updated to completed for job=%s", job_res.get("evaluation_job_id"))
            else:
                logger.warning("process_message: no matching job found to mark completed with filter=%s", job_filter)
        except Exception:
            logger.exception("process_message: failed to update jobs collection to completed (continuing)")

    except Exception:
        logger.exception("process_message: unexpected error evaluating doc %s: %s", doc_id_raw, traceback.format_exc())
        try:
            coll.update_one({"_id": doc["_id"]}, {"$set": {"evaluation_status": "error", "evaluation_error": "processing_exception", "evaluation_error_ts": datetime.now(timezone.utc)}})
        except Exception:
            logger.exception("process_message: failed to mark document as error in DB")

        # Update jobs collection -> mark job as error (best-effort)
        try:
            job_filter = {}
            job_id = message.get("evaluation_job_id") or doc.get("evaluation_job_id")
            if job_id:
                job_filter = {"evaluation_job_id": job_id}
            else:
                job_filter = {"assignment_id": str(doc.get("assignment_id")), "status": {"$in": ["in_progress", "prompt_built"]}}

            job_update = {"$set": {"status": "error", "error_ts": datetime.now(timezone.utc), "error_message": "processing_exception"}, "$push": {"history": {"event": "error", "run_id": run_id, "worker": WORKER_ID, "ts": datetime.now(timezone.utc)}}}
            job_res = jobs_coll.find_one_and_update(job_filter, job_update, return_document=ReturnDocument.AFTER)
            if job_res:
                logger.info("process_message: jobs collection updated to error for job=%s", job_res.get("evaluation_job_id"))
            else:
                logger.warning("process_message: no matching job found to mark error with filter=%s", job_filter)
        except Exception:
            logger.exception("process_message: failed to update jobs collection to error (continuing)")

    finally:
        if _MLFLOW_OK and mlflow_run_started:
            try:
                mlflow.end_run()
            except Exception:
                logger.exception("process_message: mlflow.end_run failed (continuing)")

        logger.info("process_message: exit doc=%s total_time=%.3fs", str(doc_id_raw), time.time() - start_proc)


def _compose_source_text(doc: Dict[str, Any]) -> str:
    parts = []
    system = doc.get("system_prompt") or doc.get("systemPrompt") or ""
    user = doc.get("user_prompt") or doc.get("userPrompt") or ""
    submission = doc.get("extracted_excerpt") or doc.get("extractedExcerpt") or doc.get("submission") or ""
    rubric_logic = doc.get("rubric_logic")
    if isinstance(rubric_logic, str):
        rubric_logic_str = rubric_logic
    elif rubric_logic is not None:
        try:
            import json
            rubric_logic_str = json.dumps(rubric_logic, ensure_ascii=False)
        except Exception:
            rubric_logic_str = str(rubric_logic)
    else:
        rubric_logic_str = ""
    parts.append(system)
    parts.append(user)
    parts.append(rubric_logic_str)
    parts.append(submission)
    return "\n\n".join(p.strip() for p in parts if p.strip())


def _stringify_rubric_logic(rubric_logic):
    if isinstance(rubric_logic, str):
        return rubric_logic
    elif rubric_logic is not None:
        try:
            import json
            return json.dumps(rubric_logic, ensure_ascii=False)
        except Exception:
            return str(rubric_logic)
    else:
        return ""

