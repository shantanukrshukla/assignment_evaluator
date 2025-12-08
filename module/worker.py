# module.worker.py
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
from llm_model.system_instructions import build_system_prompt_from_chunks, build_chunk_system_prompt
from utils.mlflow_init import init_mlflow

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
# Config-driven thresholds & defaults (read from settings.config.yml)
# ----------------------------
# retrieval
TOKEN_BUDGET = int(config.get("retrieval", {}).get("token_budget", 3500))
SUMMARY_THRESHOLD = int(TOKEN_BUDGET * 0.5)

# attempts / job manager
MAX_ATTEMPTS = int(config.get("job_manager", {}).get("task_retries", 1)) + 3

# relevance threshold (use retrieval.min_similarity)
RELEVANCE_THRESHOLD = float(config.get("retrieval", {}).get("min_similarity", 0.05))

# mongo collection
COLL_NAME = config.get("mongo", {}).get("prompt_collection", "prompt_builder")

# debug / prompt persistence
DEBUG_PROMPTS = bool(config.get("debug", {}).get("debug_prompts", False))
PERSIST_PROMPTS_TO_DB = bool(config.get("debug", {}).get("persist_prompts_in_db", False))
PROMPT_TRUNCATE = int(config.get("debug", {}).get("prompt_truncate_chars", 5000))

# llm section (many tunables live here)
_LLM_CFG = config.get("llm", {}) or {}
CHUNK_SUMMARY_CHARS = int(_LLM_CFG.get("chunk_summary_chars", 1000))
CHUNK_SEND_MAX_CHARS = int(_LLM_CFG.get("chunk_send_max_chars", 2000))
TOP_FULL_TEXT_MAX_CHARS = int(_LLM_CFG.get("top_full_text_max_chars", 2000))
SYSTEM_PROMPT_MAX_CHARS = int(_LLM_CFG.get("system_prompt_max_chars", 60000))
CHUNK_SYSTEM_PROMPT_MAX_CHARS = int(_LLM_CFG.get("chunk_system_prompt_max_chars", 20000))
TOP_FULL_IN_SCREENING = int(_LLM_CFG.get("top_full_in_screening", 3))

# generic defaults exposed to config
DEFAULT_MAX_CHARS = int(_LLM_CFG.get("default_max_chars", 4000))
ASSIGNMENT_MAX_CHARS = int(_LLM_CFG.get("assignment_max_chars", 6000))
SUBMISSION_MAX_CHARS = int(_LLM_CFG.get("submission_max_chars", 8000))
COMMENT_MAX_CHARS = int(_LLM_CFG.get("max_comment_chars", 2000))

# Screening and chunk schema enforcement snippets
SCREENING_SCHEMA_SNIPPET = (
    "CRITICAL: Return exactly one JSON object and adhere exactly to this schema:\n"
    '{"scores": {"<criterion_id>": 0.0-1.0, ...} | null, "overall": 0.0-1.0 | null, '
    '"feedback": "string", "violations": ["string", ...]}\n'
    "Example valid output: {\"scores\":{\"clarity\":0.8},\"overall\":0.8,\"feedback\":\"...\",\"violations\":[]}\n"
    "DO NOT include any explanatory text outside the single JSON object."
)

CHUNK_SCHEMA_SNIPPET = (
    "CRITICAL: Return exactly one JSON object and adhere exactly to this schema:\n"
    '{"chunk_id": integer, "assessments": {"<criterion_id>": 0.0-1.0, ...} | null, "comments": "string", "confidence": 0.0-1.0}\n'
    'Example valid output: {"chunk_id":0, "assessments":{"Understanding":0.8}, "comments":"...","confidence":0.8}\n'
    "DO NOT include any explanatory text outside the single JSON object."
)


# ----------------------------
# Helpers
# ----------------------------
def _compute_dedupe_key(text: str, prompt_version: str) -> str:
    h = hashlib.sha256()
    h.update((prompt_version or "").encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()


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
        t = t.replace("\u201c", "\"").replace("\u201d", "\"")
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


def _clean_and_truncate_text(raw_text: Optional[str], max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """
    Clean HTML-like text and truncate to max_chars. Default max_chars comes from config (DEFAULT_MAX_CHARS).
    """
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


def _compose_source_text(doc: Dict[str, Any]) -> str:
    parts = []
    system = doc.get("system_prompt") or doc.get("systemPrompt") or ""
    user = doc.get("user_prompt") or doc.get("userPrompt") or ""
    submission = doc.get("extracted_excerpt") or doc.get("extractedExcerpt") or doc.get("submission") or ""
    rubric_logic = (doc.get("rubric_logic") or "").strip()
    parts.append(system)
    parts.append(user)
    parts.append(rubric_logic)
    parts.append(submission)

    raw_chunks = doc.get("course_chunks") or doc.get("course_chunk") or doc.get("courseContent") or []
    for c in raw_chunks:
        if isinstance(c, dict):
            parts.append((c.get("text") or c.get("summary") or "")[:1000])
        else:
            parts.append(str(c)[:1000])
    composed = "\n".join(parts)
    logger.debug("_compose_source_text: system_present=%s user_present=%s submission_len=%d chunk_count=%d",
                 bool(system), bool(user), len(submission), len(raw_chunks))
    return composed


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
# New helper: robust int coercion for LLM-provided chunk ids
# ----------------------------
def _safe_int(val, default=None):
    """
    Safely coerce val to an int:
      - if val is already int -> return it
      - if val is numeric string like '0' or '0.0' -> int
      - if val contains digits like 'CHUNK:0' -> extract the first int
      - otherwise return default
    """
    if val is None:
        return default
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except Exception:
        pass
    try:
        return int(float(val))
    except Exception:
        pass
    try:
        s = str(val)
        m = re.search(r"-?\d+", s)
        if m:
            return int(m.group())
    except Exception:
        pass
    return default


# ----------------------------
# Chunk compacting and filtering
# ----------------------------
def _build_compact_chunks(raw_chunks: List[Any], submission_kw: List[str]) -> List[Dict[str, Any]]:
    compact = []
    for i, c in enumerate(raw_chunks):
        if isinstance(c, dict):
            text_raw = (c.get("text") or c.get("content") or "") or ""
        else:
            text_raw = str(c)

        # Use config-driven truncation for the raw text used to build summaries/keywords
        text = _clean_and_truncate_text(text_raw, max_chars=max(CHUNK_SEND_MAX_CHARS, DEFAULT_MAX_CHARS * 2))

        # Summarize if long; use chunk-summary config
        try:
            if len(text) > (CHUNK_SUMMARY_CHARS + 500):
                summary = extractive_summary(text, max_sentences=1)
            else:
                summary = text[:CHUNK_SUMMARY_CHARS]
        except Exception:
            summary = text[:CHUNK_SUMMARY_CHARS]

        keywords = top_keywords(text, topn=4)
        try:
            similarity = float(len(set(keywords) & set(submission_kw))) / (len(submission_kw) or 1)
        except Exception:
            similarity = 0.0

        compact.append({
            "id": i,
            "text": text[:CHUNK_SEND_MAX_CHARS],
            "summary": summary,
            "keywords": keywords,
            "similarity": round(similarity, 3),
            "meta": c.get("meta") if isinstance(c, dict) and c.get("meta") else {}
        })
    return compact


def _filter_chunks_by_lesson_or_chapter(raw_chunks: List[Dict[str, Any]],
                                       lesson_id: Optional[int] = None,
                                       chapter_id: Optional[int] = None) -> List[Dict[str, Any]]:
    if not raw_chunks:
        return []

    matched = []
    for c in raw_chunks:
        try:
            if not isinstance(c, dict):
                continue
            meta = c.get("meta") or {}
            top_lesson = c.get("lesson_id")
            if lesson_id is not None and _is_number_like(top_lesson) and int(top_lesson) == int(lesson_id):
                matched.append(c)
                continue
            lesson_ids = meta.get("lesson_ids") or []
            if lesson_id is not None and isinstance(lesson_ids, (list, tuple)) and any(
                    _is_number_like(x) and int(x) == int(lesson_id) for x in lesson_ids):
                matched.append(c)
                continue
            meta_chap = meta.get("chapter_id")
            if chapter_id is not None and _is_number_like(meta_chap) and int(meta_chap) == int(chapter_id):
                matched.append(c)
                continue
            if lesson_id is not None and _is_number_like(meta_chap) and int(meta_chap) == int(lesson_id):
                matched.append(c)
                continue
        except Exception:
            continue
    return matched


# ----------------------------
# Core evaluation flow
# ----------------------------
def evaluate_document(doc: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
    start_all = time.time()
    coll = db[COLL_NAME]
    doc_id = str(doc.get("_id"))
    assignment_id = doc.get("assignment_id")
    logger.info("evaluate_document: start doc=%s assignment=%s", doc_id, assignment_id)

    prompt_version = doc.get("prompt_version") or "v1"

    # Normalize course payload into raw_chunks if necessary
    raw_chunks = doc.get("course_chunks") or doc.get("course_chunk") or doc.get("courseContent") or []
    if (not raw_chunks) and doc.get("course_payload"):
        try:
            normalized = normalize_course_payload(doc.get("course_payload") or {})
            rc = []
            for ch in normalized.get("chapters", []):
                for lesson in ch.get("lessons", []):
                    # Prefer explicit 'transcript' if present in normalized payload.
                    lesson_transcript = lesson.get("transcript") or lesson.get("text") or ""
                    rc.append({
                        "lesson_id": lesson.get("lesson_id"),
                        "text": lesson_transcript,
                        "transcript": lesson_transcript,
                        "meta": {
                            "chapter_title": ch.get("chapter_title"),
                            "lesson_title": lesson.get("lesson_title"),
                            **(lesson.get("meta") or {})
                        }
                    })
            if rc:
                raw_chunks = rc
                logger.info("evaluate_document: expanded course_payload to %d raw_chunks for assignment=%s", len(rc),
                            assignment_id)
            else:
                logger.warning("evaluate_document: course_payload present but yielded NO raw_chunks for assignment=%s",
                               assignment_id)
        except Exception:
            logger.exception("evaluate_document: failed to normalize_course_payload for %s", assignment_id)

    if not raw_chunks:
        logger.warning(
            "evaluate_document: raw_chunks is EMPTY for doc=%s assignment=%s. Prompt will lack course context!",
            doc_id, assignment_id)
    else:
        logger.info("evaluate_document: raw_chunks count=%d for doc=%s", len(raw_chunks), doc_id)

    # Extract submission text and summary/keywords
    submission = doc.get("extracted_excerpt") or doc.get("extractedExcerpt") or doc.get("submission") or ""
    if submission:
        logger.info("evaluate_document: found submission text (len=%d)", len(submission))
    else:
        logger.warning("evaluate_document: submission text is EMPTY for doc=%s", doc_id)
    if len(submission) > SUMMARY_THRESHOLD:
        submission_summary = extractive_summary(submission, max_sentences=2)
    else:
        submission_summary = submission[:SUBMISSION_MAX_CHARS]
    submission_kw = top_keywords(submission, topn=6)

    # Detect lesson_id/chapter_id hints and filter chunks if present
    lesson_id = None
    chapter_id = None
    try:
        if doc.get("lesson_id") is not None:
            lesson_id = int(doc.get("lesson_id"))
        elif message.get("lesson_id") is not None:
            lesson_id = int(message.get("lesson_id"))
    except Exception:
        lesson_id = None

    try:
        if doc.get("chapter_id") is not None:
            chapter_id = int(doc.get("chapter_id"))
        elif message.get("chapter_id") is not None:
            chapter_id = int(message.get("chapter_id"))
    except Exception:
        chapter_id = None

    if lesson_id is not None or chapter_id is not None:
        logger.info("evaluate_document: filtering raw_chunks by lesson_id=%s chapter_id=%s", str(lesson_id), str(chapter_id))
        filtered = _filter_chunks_by_lesson_or_chapter(raw_chunks, lesson_id=lesson_id, chapter_id=chapter_id)
        if filtered:
            logger.info("evaluate_document: filtered chunks count=%d (from %d)", len(filtered), len(raw_chunks))
            raw_chunks = filtered
        else:
            logger.warning("evaluate_document: no chunks matched lesson_id=%s chapter_id=%s; keeping original chunks",
                           lesson_id, chapter_id)
    else:
        logger.debug("evaluate_document: no lesson_id/chapter_id provided; not filtering course chunks")

    # Debug: log matched chunk ids and transcript availability
    try:
        if raw_chunks:
            for i, rc in enumerate(raw_chunks):
                if isinstance(rc, dict):
                    logger.debug("evaluate_document: post-filter chunk idx=%s lesson_id=%s meta_chapter=%s has_transcript=%s transcript_len=%d",
                                 i, rc.get("lesson_id"), (rc.get("meta") or {}).get("chapter_id"),
                                 bool(rc.get("transcript")), len((rc.get("transcript") or "")))
                else:
                    logger.debug("evaluate_document: post-filter chunk idx=%s (non-dict)", i)
    except Exception:
        logger.exception("evaluate_document: debug logging failed for post-filter chunks")

    # Build compact chunks once
    raw_chunks = raw_chunks or []
    compact_chunks = _build_compact_chunks(raw_chunks, submission_kw)
    logger.debug("evaluate_document: compact_chunks_count=%s", len(compact_chunks))
    if raw_chunks and not compact_chunks:
        logger.error(
            "evaluate_document: raw_chunks existed (%d) but compact_chunks is EMPTY! Check _build_compact_chunks logic.",
            len(raw_chunks))

    # NEW: compute max similarity between submission and chunks (0.0 - 1.0)
    try:
        max_chunk_similarity = 0.0
        if compact_chunks:
            max_chunk_similarity = max([c.get("similarity", 0.0) for c in compact_chunks] or [0.0])
        logger.debug("evaluate_document: max_chunk_similarity=%s (threshold=%s)", max_chunk_similarity, RELEVANCE_THRESHOLD)
    except Exception:
        max_chunk_similarity = 0.0

    # Compose dedupe key
    source_text = _compose_source_text(doc)
    dedupe_key = _compute_dedupe_key(source_text, prompt_version)
    logger.info("evaluate_document: dedupe_key=%s doc=%s", dedupe_key[:12], doc_id)

    # Prepare top-full-texts to include in screening
    try:
        top_full_cfg = int(config.get("llm", {}).get("top_full_in_screening", TOP_FULL_IN_SCREENING))
    except Exception:
        top_full_cfg = TOP_FULL_IN_SCREENING
    TOP_FULL_COUNT = min(3, max(0, int(top_full_cfg)))
    sorted_by_sim = sorted(compact_chunks, key=lambda x: x.get("similarity", 0.0), reverse=True)
    top_ids = [c["id"] for c in sorted_by_sim[:TOP_FULL_COUNT]]

    if not top_ids and raw_chunks:
        top_ids = list(range(min(TOP_FULL_COUNT, len(raw_chunks))))
        logger.debug("evaluate_document: no similarity hits; falling back to first chunk ids=%s for doc=%s", top_ids,
                     doc_id)

    top_full_texts = []
    for cid in top_ids:
        try:
            if isinstance(cid, int) and cid < len(raw_chunks):
                raw_chunk = raw_chunks[cid]
            else:
                raw_chunk = None
            if isinstance(raw_chunk, dict):
                # Prefer transcript, then text, then content, then summary
                ft_src = (
                    raw_chunk.get("transcript")
                    or raw_chunk.get("text")
                    or raw_chunk.get("content")
                    or raw_chunk.get("summary")
                    or ""
                )
                ft = _clean_and_truncate_text(ft_src, max_chars=TOP_FULL_TEXT_MAX_CHARS)
                logger.debug(
                    "evaluate_document: top_full_texts prepared cid=%s len=%d used_field=%s",
                    cid, len(ft), "transcript" if raw_chunk.get("transcript") else (
                        "text" if raw_chunk.get("text") else (
                            "content" if raw_chunk.get("content") else (
                                "summary" if raw_chunk.get("summary") else "none"
                            )
                        )
                    )
                )
            else:
                ft = _clean_and_truncate_text(str(raw_chunk), max_chars=TOP_FULL_TEXT_MAX_CHARS)
                logger.debug("evaluate_document: top_full_texts prepared cid=%s len=%d used_field=str(raw_chunk)", cid, len(ft))
            top_full_texts.append({"id": cid, "text": ft})
        except Exception:
            logger.exception("evaluate_document: failed to prepare top_full_text for cid=%s", cid)
            continue

    # Build system prompt using rubric_logic in place of rubric (screening uses SCREENING schema)
    try:
        rubric_logic = (doc.get("rubric_logic") or "").strip()
        system_prompt = build_system_prompt_from_chunks(
            compact_chunks=compact_chunks,
            top_full_texts=top_full_texts,
            rubric=rubric_logic,  # pass rubric_logic in place of rubric
            course_id=doc.get("course_id") or doc.get("assignment_id"),
            max_context_chars=SYSTEM_PROMPT_MAX_CHARS
        )
        if rubric_logic:
            # ensure rubric_logic is visible first
            system_prompt = rubric_logic + "\n\n" + system_prompt
        # enforce screening schema at the top of the system prompt
        system_prompt = SCREENING_SCHEMA_SNIPPET + "\n\n" + system_prompt
        logger.info("evaluate_document: built system_prompt len=%d for doc=%s", len(system_prompt), doc_id)
    except Exception:
        logger.exception(
            "evaluate_document: failed to build system_prompt from chunks; falling back to doc.system_prompt")
        fallback_sys = (doc.get("system_prompt") or "You are an automated evaluator.")
        rubric_logic = (doc.get("rubric_logic") or "").strip()
        if rubric_logic:
            fallback_sys = rubric_logic + "\n\n" + fallback_sys
        system_prompt = SCREENING_SCHEMA_SNIPPET + "\n\n" + fallback_sys

    # Build user prompt: assignment description (if present) + submission
    assignment_text = doc.get("assignment_text") or doc.get("assignment_description") or doc.get("assignment") or ""
    if assignment_text:
        user_prompt = (
            "Assignment (context for evaluation):\n\n"
            + _clean_and_truncate_text(assignment_text, max_chars=ASSIGNMENT_MAX_CHARS)
            + "\n\nStudent submission:\n\n"
            + _clean_and_truncate_text(submission, max_chars=SUBMISSION_MAX_CHARS)
        )
    else:
        user_prompt = _clean_and_truncate_text(submission, max_chars=SUBMISSION_MAX_CHARS)

    # Persist a lightweight debug snapshot (best-effort)
    if DEBUG_PROMPTS or PERSIST_PROMPTS_TO_DB:
        try:
            dbg = {"compact_chunks_sample": compact_chunks[:6], "top_full_texts_ids": [t["id"] for t in top_full_texts]}
            coll.update_one({"_id": doc.get("_id")}, {"$push": {
                "debug.prompts": {"type": "pre_system_snapshot", "payload": dbg, "ts": datetime.now(timezone.utc)}}})
        except Exception:
            logger.exception("evaluate_document: failed to persist pre_system_snapshot to DB")

    # Log prompt lengths
    logger.info("evaluate_document: screening_prompt_len=%d system_prompt_len=%d user_prompt_len=%d",
                len(json.dumps({
                    "compact_chunks": compact_chunks,
                    "top_full_texts": top_full_texts
                }, ensure_ascii=False)),
                len(system_prompt), len(user_prompt))
    if DEBUG_PROMPTS:
        logger.debug("DEBUG_SYSTEM_PROMPT_PREVIEW (trunc %d): %s", PROMPT_TRUNCATE, system_prompt[:PROMPT_TRUNCATE])
        logger.debug("DEBUG_USER_PROMPT_PREVIEW (trunc %d): %s", PROMPT_TRUNCATE, (user_prompt or "")[:PROMPT_TRUNCATE])

    # Call LLM for screening
    t_screen_start = time.time()
    # provider override: allow message or document to override global config for this run
    provider_override = (message.get("llm_provider") or doc.get("llm_provider") or None)
    if provider_override:
        logger.info("evaluate_document: using provider_override=%s for doc=%s", provider_override, doc_id)
    screening_raw = ask_llm(system_prompt=system_prompt, user_prompt=user_prompt, provider_override=provider_override)
    t_screen_end = time.time()
    logger.info("evaluate_document: screening LLM returned duration=%.3fs raw_len=%s", (t_screen_end - t_screen_start),
                len(screening_raw) if screening_raw else 0)
    if DEBUG_PROMPTS:
        logger.debug("DEBUG_SCREENING_RAW (trunc 2000): %s", (screening_raw or "")[:2000])

    # Try parse screening output (screening expected to follow SCREENING schema)
    screening_json = _safe_parse_json(screening_raw) or {}
    final_obj = screening_json.get("final") or {}
    if not final_obj and any(k in screening_json for k in ("overall", "scores", "feedback", "violations")):
        final_obj = screening_json

    normalized_final = _validate_and_normalize_final(final_obj, rubric_logic=rubric_logic)
    if normalized_final is None:
        logger.warning("evaluate_document: screening final_obj not valid; attempting reformat retry for doc=%s", doc_id)
        reformat_system = (
            "You are a strict JSON formatter. You WILL output exactly one JSON object and nothing else. "
            "Do NOT include any explanation or commentary."
        )
        reformat_user_template = (
            "Please extract numeric scores and overall from the following assistant output and return exactly one JSON object conforming to:\n"
            '{"scores": {"<criterion_id>": 0.0-1.0, ...} | null, "overall": 0.0-1.0 | null, "feedback": "string", "violations": ["string", ...]}\n\n'
            "If numeric scores use another scale (e.g. 0-5, 0-10, 0-100), rescale to 0.0-1.0. "
            "If you cannot extract per-criterion scores, set scores to null and overall to null.\n\n"
            "Original assistant output:\n\n{ORIGINAL}\n\nReturn only the JSON object."
        )
        reformat_user = reformat_user_template.replace("{ORIGINAL}", screening_raw or "")

        try:
            reformat_raw = ask_llm(system_prompt=reformat_system, user_prompt=reformat_user, provider_override=provider_override)
            if DEBUG_PROMPTS:
                logger.debug("DEBUG_REFORMAT_RAW_1 (trunc 2000): %s", (reformat_raw or "")[:2000])
            reformat_json = _safe_parse_json(reformat_raw) or {}
            reformat_final = reformat_json.get("final") or reformat_json
            normalized_final = _validate_and_normalize_final(reformat_final, rubric_logic=rubric_logic)
            if normalized_final:
                logger.info("evaluate_document: reformat retry 1 succeeded for doc=%s", doc_id)
        except Exception:
            logger.exception("evaluate_document: reformat retry 1 failed for doc=%s", doc_id)

        if normalized_final is None:
            try:
                reformat_user_2 = reformat_user_template.replace("{ORIGINAL}", screening_raw or "") + \
                                "\n\nIf criteria names are ambiguous, map the numeric scores to rubric logic where reasonable; otherwise set scores to null."
                reformat_raw_2 = ask_llm(system_prompt=reformat_system, user_prompt=reformat_user_2, provider_override=provider_override)
                if DEBUG_PROMPTS:
                    logger.debug("DEBUG_REFORMAT_RAW_2 (trunc 2000): %s", (reformat_raw_2 or "")[:2000])
                reformat_json_2 = _safe_parse_json(reformat_raw_2) or {}
                reformat_final_2 = reformat_json_2.get("final") or reformat_json_2
                normalized_final = _validate_and_normalize_final(reformat_final_2, rubric_logic=rubric_logic)
                if normalized_final:
                    logger.info("evaluate_document: reformat retry 2 succeeded for doc=%s", doc_id)
                else:
                    logger.warning("evaluate_document: reformat retry 2 ALSO failed for doc=%s", doc_id)
            except Exception:
                logger.exception("evaluate_document: reformat retry 2 failed for doc=%s", doc_id)

    priority_chunks = screening_json.get("priority_chunks") or []

    # Normalize priority_chunks into integer ids where possible.
    norm_priority = []
    if priority_chunks:
        for cid in priority_chunks:
            n = _safe_int(cid, default=None)
            if n is None:
                logger.debug("evaluate_document: dropping non-numeric priority_chunk id=%s", cid)
                continue
            norm_priority.append(n)

    # If normalization produced nothing, fall back to similarity-based ids
    if not norm_priority:
        sorted_by_sim = sorted(compact_chunks, key=lambda x: x.get("similarity", 0.0), reverse=True)
        norm_priority = [c["id"] for c in sorted_by_sim[:min(3, len(sorted_by_sim))]]
        logger.debug("evaluate_document: fallback priority_chunks=%s", norm_priority)

    priority_chunks = norm_priority

    if not priority_chunks:
        priority_chunks = []

    screening_obj = screening_json.get("screening") or {}

    # ENFORCE irrelevance deterministically based on max_chunk_similarity:
    forced_irrelevant = False
    if max_chunk_similarity < RELEVANCE_THRESHOLD:
        logger.info(
            "evaluate_document: submission considered IRRELEVANT (max_similarity=%.4f < threshold=%.4f) for doc=%s",
            max_chunk_similarity, RELEVANCE_THRESHOLD, doc_id
        )
        forced_irrelevant = True
        zero_scores = {}
        if rubric_logic:
            candidate_keys = []
            for key in ["Understanding of the Topic", "Correctness & Accuracy",
                        "Depth of Reasoning & Explanation", "Structure, Organization & Clarity",
                        "Completeness & Requirement Coverage", "Understanding", "Correctness", "Depth", "Structure", "Completeness"]:
                if key in rubric_logic:
                    candidate_keys.append(key)
            if candidate_keys:
                for k in candidate_keys:
                    zero_scores[k] = 0.0
        if not zero_scores:
            zero_scores = {
                "Understanding of the Topic": 0.0,
                "Correctness & Accuracy": 0.0,
                "Depth of Reasoning & Explanation": 0.0,
                "Structure, Organization & Clarity": 0.0,
                "Completeness & Requirement Coverage": 0.0
            }
        normalized_final = {
            "scores": {k: 0.0 for k in zero_scores.keys()},
            "overall": 0.0,
            "feedback": "Submission is irrelevant to the provided lesson chunk(s); evaluated as 0/100 per irrelevance rule.",
            "violations": ["irrelevant_submission"]
        }
        # prepare chunk_results marking irrelevance for traceability
        chunk_results = []
        for cid in (priority_chunks or []):
            chunk_id_safe = _safe_int(cid, default=None)
            chunk_results.append({
                "chunk_id": chunk_id_safe if chunk_id_safe is not None else cid,
                "assessments": {k: 0.0 for k in zero_scores.keys()},
                "comments": "Submission unrelated to this chunk (irrelevant).",
                "confidence": 0.0
            })
    else:
        chunk_results = []

    # Per-chunk evaluation (if not forced_irrelevant)
    chunk_raw_responses = []
    if not forced_irrelevant:
        for cid in priority_chunks:
            chunk_start = time.time()
            try:
                if isinstance(cid, int) and cid < len(raw_chunks):
                    raw_chunk = raw_chunks[cid]
                else:
                    raw_chunk = None
                orig_text = raw_chunk.get("text") if isinstance(raw_chunk, dict) else (str(raw_chunk) if raw_chunk else "")
            except Exception:
                orig_text = ""

            # Use config-driven chunk send behaviour to preserve original intent
            try:
                threshold_for_full = CHUNK_SEND_MAX_CHARS * 2
            except Exception:
                threshold_for_full = DEFAULT_MAX_CHARS * 2
            send_text = orig_text if len(orig_text) <= threshold_for_full else _clean_and_truncate_text(orig_text, max_chars=CHUNK_SEND_MAX_CHARS)

            try:
                chunk_system_prompt = build_chunk_system_prompt(
                    chunk_id=cid,
                    chunk_text=_clean_and_truncate_text(send_text, max_chars=CHUNK_SYSTEM_PROMPT_MAX_CHARS),
                    rubric=rubric_logic,
                    course_id=doc.get("course_id") or doc.get("assignment_id"),
                    max_context_chars=CHUNK_SYSTEM_PROMPT_MAX_CHARS
                )
                chunk_system_prompt = CHUNK_SCHEMA_SNIPPET + "\n\n" + chunk_system_prompt
            except Exception:
                logger.exception("evaluate_document: failed to build chunk_system_prompt for cid=%s; falling back", cid)
                chunk_system_prompt = CHUNK_SCHEMA_SNIPPET + "\n\n" + (doc.get("system_prompt") or "You are an evaluator for a course chunk.")

            chunk_payload = {
                "assignment_id": assignment_id,
                "student_id": doc.get("student_id"),
                "chunk": {
                    "chunk_id": cid,
                    "text": send_text[:CHUNK_SEND_MAX_CHARS]
                },
                "submission_summary": submission_summary,
                "rubric_logic": rubric_logic or "",
                "instruction": (
                    "Evaluate the student's submission RELATIVE TO THIS CHUNK only. "
                    "Provide a JSON object with keys: "
                    "  - chunk_id: integer, "
                    "  - assessments: object (optional rubric breakdown), "
                    "  - comments: string (concise feedback), "
                    "  - confidence: float 0.0-1.0"
                ),
                "notes": {"max_comment_chars": COMMENT_MAX_CHARS}
            }
            focus_prompt_text = json.dumps(chunk_payload, ensure_ascii=False)

            logger.info("evaluate_document: calling LLM for chunk cid=%s prompt_len=%d", str(cid), len(focus_prompt_text))
            if DEBUG_PROMPTS:
                logger.debug("DEBUG_FULL_CHUNK_PROMPT cid=%s (trunc %d): %s", cid, PROMPT_TRUNCATE, focus_prompt_text[:PROMPT_TRUNCATE])

            t_chunk_start = time.time()
            resp_text = ask_llm(system_prompt=chunk_system_prompt, user_prompt=focus_prompt_text, provider_override=provider_override)
            t_chunk_end = time.time()
            logger.info("evaluate_document: chunk LLM finished cid=%s duration=%.3fs resp_len=%s", str(cid),
                        (t_chunk_end - t_chunk_start), len(resp_text) if resp_text else 0)
            if DEBUG_PROMPTS:
                logger.debug("DEBUG_CHUNK_RAW cid=%s (trunc 2000): %s", cid, (resp_text or "")[:2000])

            # store raw chunk response for traceability
            chunk_raw_responses.append({"cid": cid, "raw": resp_text})

            parsed = _safe_parse_json(resp_text)
            if not parsed:
                logger.warning("evaluate_document: chunk parse failed for cid=%s; storing parse_error_or_no_response", str(cid))
                parsed = {"chunk_id": cid, "assessments": {}, "comments": "parse_error_or_no_response", "confidence": 0.0}
            else:
                try:
                    parsed_cid = parsed.get("chunk_id", cid)
                    parsed_int = _safe_int(parsed_cid, default=None)
                    parsed["chunk_id"] = parsed_int if parsed_int is not None else parsed_cid
                except Exception:
                    parsed["chunk_id"] = cid
                try:
                    c = float(parsed.get("confidence", 0.0))
                    parsed["confidence"] = max(0.0, min(1.0, c))
                except Exception:
                    parsed["confidence"] = 0.0

                assessments = parsed.get("assessments", {})
                norm_assess = {}
                if assessments is None:
                    parsed["assessments"] = None
                else:
                    for k, v in (assessments or {}).items():
                        try:
                            num = float(v)
                            num = max(0.0, min(1.0, num))
                            norm_assess[k] = round(num, 4)
                        except Exception:
                            pass
                    parsed["assessments"] = norm_assess

            chunk_results.append(parsed)
            logger.debug("evaluate_document: chunk cid=%s elapsed=%.3fs", str(cid), time.time() - chunk_start)

    # If normalized_final was not set previously (e.g., forced_irrelevant set it), ensure we pull from it:
    try:
        normalized_scores = normalized_final.get("scores") if normalized_final else None
        overall = normalized_final.get("overall") if normalized_final else None
        feedback_text = normalized_final.get("feedback") if normalized_final else (final_obj.get("feedback") or "")
        violations = normalized_final.get("violations") if normalized_final else (final_obj.get("violations") or [])
    except Exception:
        normalized_scores = None
        overall = None
        feedback_text = ""
        violations = []

    model_raw = {
        "screening_raw": screening_raw,
        "screening_json": screening_json,
        "chunk_raw_responses": chunk_raw_responses or [{"cid": c.get("chunk_id"), "raw": None} for c in chunk_results]
    }

    result = {
        "scores": normalized_scores if normalized_scores else None,
        "overall": overall,
        "feedback": feedback_text,
        "violations": violations,
        "chunk_results": chunk_results,
        "screening_raw": screening_raw,
        "screening_json": screening_json,
        "dedupe_key": dedupe_key,
        "prompt_version": prompt_version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model_raw_response": model_raw
    }

    logger.info(
        "evaluate_document: finished doc=%s assignment=%s total_time=%.3fs chunk_count=%s",
        doc_id, assignment_id, time.time() - start_all, len(chunk_results)
    )

    # MLflow logging (best-effort) -- DOES NOT start a run here; start occurs in process_message
    if _MLFLOW_OK:
        try:
            mlflow.set_tag("assignment_id", assignment_id or "unknown")
            mlflow.set_tag("doc_id", doc_id)
            if overall is not None:
                mlflow.log_metric("evaluation_overall", float(overall))
            mlflow.log_metric("evaluation_chunk_count", len(chunk_results))
            # log per-criterion metrics if available
            if normalized_scores:
                for k, v in normalized_scores.items():
                    try:
                        # sanitize metric name
                        metric_name = "score_" + re.sub(r"[^a-zA-Z0-9_]", "_", str(k).lower())
                        mlflow.log_metric(metric_name, float(v))
                    except Exception:
                        continue
        except Exception:
            logger.exception("evaluate_document: mlflow logging failed (continuing)")

    return result


# ----------------------------
# Process message (entrypoint)
# ----------------------------
def process_message(message: Dict[str, Any]):
    from pymongo import ReturnDocument
    coll = db[COLL_NAME]

    # jobs collection (configured in config.yml)
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
    source_text = _compose_source_text(doc)
    dedupe_key = _compute_dedupe_key(source_text, prompt_version)
    existing_dk = doc.get("dedupe_key")
    logger.debug("process_message: computed dedupe_key=%s existing=%s", dedupe_key[:12], (existing_dk or "")[:12])

    if existing_dk == dedupe_key and doc.get("evaluated") and not force:
        logger.info("process_message: skipping evaluation for %s (dedupe match)", doc_id_raw)
        coll.update_one({"_id": doc["_id"]}, {"$push": {
            "evaluation_job_history": {"run_id": str(time.time()), "status": "skipped", "dedupe_key": dedupe_key,
                                       "timestamp": datetime.now(timezone.utc)}}})
        return

    # Attempt an atomic claim using find_one_and_update with retries to reduce "pre-claim failed" races
    claim_attempts = int(config.get("job_manager", {}).get("claim_retries", 3))
    claim_delay = float(config.get("job_manager", {}).get("claim_backoff_seconds", 0.5))
    claimed_doc = None

    # If skip_claim is present we behave like a forced takeover: attempt to set status unconditionally
    if message.get("skip_claim"):
        try:
            res = coll.find_one_and_update(
                {"_id": doc["_id"]},
                {"$set": {
                    "evaluation_status": "in_progress",
                    "evaluation_started_at": datetime.now(timezone.utc),
                    "evaluation_worker_id": WORKER_ID
                }, "$inc": {"evaluation_attempts": 1}},
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
        # Normal atomic claim: only claim if status is absent or queued or error
        claim_filter = {
            "_id": doc["_id"],
            "$or": [
                {"evaluation_status": {"$exists": False}},
                {"evaluation_status": "queued"},
                {"evaluation_status": "error"}
            ]
        }
        claim_update = {
            "$set": {
                "evaluation_status": "in_progress",
                "evaluation_started_at": datetime.now(timezone.utc),
                "evaluation_worker_id": WORKER_ID
            },
            "$inc": {"evaluation_attempts": 1}
        }

        for attempt in range(1, max(1, claim_attempts) + 1):
            try:
                res = coll.find_one_and_update(
                    claim_filter,
                    claim_update,
                    return_document=ReturnDocument.AFTER,
                )
                if res:
                    claimed_doc = res
                    logger.info("process_message: atomic claim succeeded for doc=%s on attempt=%d", doc_id_raw, attempt)
                    break
                else:
                    logger.info("process_message: atomic claim attempt=%d failed for doc=%s (likely claimed by another worker)", attempt, doc_id_raw)
            except Exception:
                logger.exception("process_message: atomic claim attempt=%d raised exception for doc=%s", attempt, doc_id_raw)

            # small backoff before retry (bounded)
            if attempt < claim_attempts:
                try:
                    time.sleep(claim_delay * attempt)
                except Exception:
                    pass

    if not claimed_doc:
        # Still not claimed after retries — log and return (consistent with previous behavior)
        logger.info("process_message: skip %s (pre-claim failed after %d attempts)", str(doc_id_raw), claim_attempts)
        coll.update_one({"_id": doc["_id"]}, {"$push": {
            "evaluation_job_history": {"run_id": str(time.time()), "status": "pre_claim_failed",
                                       "attempts": claim_attempts, "timestamp": datetime.now(timezone.utc)}}})
        return

    run_id = str(int(time.time() * 1000))
    run_entry = {"run_id": run_id, "requested_by": requester, "requested_at": datetime.now(timezone.utc),
                 "status": "in_progress", "dedupe_key": dedupe_key, "prompt_version": prompt_version}
    logger.info("process_message: claimed doc _id=%s run_id=%s worker_id=%s", str(doc["_id"]), run_id, WORKER_ID)

    # ---------------------------
    # Update jobs collection -> mark job as in_progress (best-effort)
    # ---------------------------
    try:
        job_filter = {}
        job_id = message.get("evaluation_job_id") or doc.get("evaluation_job_id")
        if job_id:
            job_filter = {"evaluation_job_id": job_id}
        else:
            # fallback: match by assignment_id and expected prior status (e.g., prompt_built)
            job_filter = {"assignment_id": str(doc.get("assignment_id")), "status": "prompt_built"}

        job_update = {
            "$set": {
                "status": "in_progress",
                "started_by": WORKER_ID,
                "started_at": datetime.now(timezone.utc)
            },
            "$push": {"history": {"event": "claimed", "run_id": run_id, "worker": WORKER_ID, "ts": datetime.now(timezone.utc)}}
        }
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

    mlflow_run_started = False
    if _MLFLOW_OK:
        try:
            mlflow.start_run(run_name=run_id)
            mlflow.set_tag("run_id", run_id)
            mlflow.set_tag("worker_id", WORKER_ID)
            mlflow.log_param("assignment_id", doc.get("assignment_id") or "unknown")
            mlflow_run_started = True
        except Exception:
            logger.exception("process_message: mlflow.start_run failed (continuing)")
            mlflow_run_started = False

    try:
        eval_start = time.time()
        eval_result = evaluate_document(doc, message)
        eval_duration = time.time() - eval_start
        logger.info("process_message: evaluation complete for doc=%s run_id=%s duration=%.3fs",
                    str(doc.get("_id") or doc.get("assignment_id") or "<unknown>"), run_id, eval_duration)

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

        model_raw = eval_result.get("model_raw_response") or {"screening_raw": eval_result.get("screening_raw"),
                                                              "chunk_results": eval_result.get("chunk_results")}

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
                      "dedupe_key": dedupe_key}}
        )
        if getattr(ok, "modified_count", 0) == 0:
            logger.warning("process_message: final update race for doc=%s; appending history only", str(doc["_id"]))
            coll.update_one({"_id": doc["_id"]}, {"$push": {"evaluation_job_history": run_entry}})
        else:
            logger.info("process_message: final update succeeded for doc=%s", str(doc["_id"]))

        _append_memory_for_doc(coll, doc["_id"], eval_result.get("scores"), eval_result.get("feedback"),
                               eval_result.get("chunk_results"))

        # ---------------------------
        # Update jobs collection -> mark job as completed (best-effort)
        # ---------------------------
        try:
            job_filter = {}
            job_id = message.get("evaluation_job_id") or doc.get("evaluation_job_id")
            if job_id:
                job_filter = {"evaluation_job_id": job_id}
            else:
                job_filter = {"assignment_id": str(doc.get("assignment_id")), "status": "in_progress"}

            job_update = {
                "$set": {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc),
                    "duration_seconds": round(eval_duration, 3),
                    "result_summary": {
                        "overall": eval_result.get("overall"),
                        "scores": eval_result.get("scores")
                    }
                },
                "$push": {"history": {"event": "completed", "run_id": run_id, "worker": WORKER_ID, "ts": datetime.now(timezone.utc)}}
            }
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
            coll.update_one({"_id": doc["_id"]}, {
                "$set": {"evaluation_status": "error", "evaluation_error": "processing_exception",
                         "evaluation_error_ts": datetime.now(timezone.utc)}})
        except Exception:
            logger.exception("process_message: failed to mark document as error in DB")

        # ---------------------------
        # Update jobs collection -> mark job as error (best-effort)
        # ---------------------------
        try:
            job_filter = {}
            job_id = message.get("evaluation_job_id") or doc.get("evaluation_job_id")
            if job_id:
                job_filter = {"evaluation_job_id": job_id}
            else:
                job_filter = {"assignment_id": str(doc.get("assignment_id")), "status": {"$in": ["in_progress", "prompt_built"]}}

            job_update = {
                "$set": {
                    "status": "error",
                    "error_ts": datetime.now(timezone.utc),
                    "error_message": "processing_exception"
                },
                "$push": {"history": {"event": "error", "run_id": run_id, "worker": WORKER_ID, "ts": datetime.now(timezone.utc)}}
            }
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
