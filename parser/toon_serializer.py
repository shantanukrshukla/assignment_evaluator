# parser/toon_serializer.py
"""
TOON serializer that prefers the external `python-toon` package when available.
If python-toon is not installed or its API is incompatible, we fall back
to a deterministic compact textual representation (TOONv1 lines).
"""
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Short key mapping (extend as needed)
_KEY_MAP = {
    "student_submission": "S",
    "course_chunks": "C",
    "system_prompt": "SYS",
    "user_prompt": "USR",
    "rubric": "RUB",
    "assignment_id": "A",
    "student_id": "S_ID",
}

def minify_keys(obj):
    """
    Recursively replace known long keys with mapped short tokens.
    Returns a new structure (doesn't modify input).
    Useful to improve determinism & compactness when serializing to TOON.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = _KEY_MAP.get(k, k)
            out[nk] = minify_keys(v)
        return out
    if isinstance(obj, list):
        return [minify_keys(x) for x in obj]
    return obj

def _escape_line(text: str) -> str:
    # minimal escaping to avoid newline injection; keep compact
    return text.replace("\n", " ").replace("\r", " ").strip()

# --- Try to import python-toon (multiple common names) ---
_toon_lib = None
_toon_name = None
for modname in ("toon", "python_toon", "python_toonlib", "pytoon", "toonlib"):
    try:
        _m = __import__(modname)
        # Heuristic: if lib exposes a dumps/encode/to_toon function we can call
        if any(hasattr(_m, fn) for fn in ("dumps", "encode", "to_toon", "dump_to_toon", "serialize")):
            _toon_lib = _m
            _toon_name = modname
            logger.info("parser.toon_serializer: using external python-toon module: %s", modname)
            break
    except Exception:
        # ignore import errors and try next
        continue

def _use_external_toon(obj: Dict[str, Any]) -> Optional[str]:
    """
    Try to convert obj -> toon textual representation via external lib.
    Returns the string or None on failure.
    """
    if _toon_lib is None:
        return None
    try:
        # Try common method names in order of likelihood.
        if hasattr(_toon_lib, "dumps"):
            out = _toon_lib.dumps(obj)
            if isinstance(out, bytes):
                return out.decode("utf-8", errors="replace")
            return str(out)
        if hasattr(_toon_lib, "encode"):
            out = _toon_lib.encode(obj)
            if isinstance(out, bytes):
                return out.decode("utf-8", errors="replace")
            return str(out)
        if hasattr(_toon_lib, "to_toon"):
            out = _toon_lib.to_toon(obj)
            if isinstance(out, bytes):
                return out.decode("utf-8", errors="replace")
            return str(out)
        if hasattr(_toon_lib, "dump_to_toon"):
            out = _toon_lib.dump_to_toon(obj)
            if isinstance(out, bytes):
                return out.decode("utf-8", errors="replace")
            return str(out)
        # last resort: try json -> library encoder if it exposes generic serialize
        if hasattr(_toon_lib, "serialize"):
            out = _toon_lib.serialize(obj)
            if isinstance(out, bytes):
                return out.decode("utf-8", errors="replace")
            return str(out)
    except Exception as exc:
        logger.warning("parser.toon_serializer: external python-toon conversion failed: %s", exc)
    return None

def build_compact_toon_text(doc_id: str, prompt_version: str, meta: Dict[str, Any],
                            submission_summary: str, submission_kw: List[str],
                            chunks: List[Dict[str, Any]], rubric: Optional[Dict[str, Any]] = None) -> str:
    """
    Build a compact line-based TOONv1 textual block.
    Designed to be token-efficient when sent as a single prompt string.
    """
    lines = []
    header = f"TOONv1|id={doc_id}|pv={prompt_version}"
    lines.append(header)

    # meta
    m_fields = []
    if meta:
        if meta.get("assignment_id"): m_fields.append(f"A={meta.get('assignment_id')}")
        if meta.get("student_id"): m_fields.append(f"S_ID={meta.get('student_id')}")
    if m_fields:
        lines.append("M|" + "|".join(m_fields))

    # rubric
    if rubric:
        try:
            rub_entries = []
            if isinstance(rubric, dict):
                for k, v in rubric.items():
                    rub_entries.append(f"{k}:{v}")
            else:
                rub_entries.append(str(rubric))
            lines.append("R|" + ";".join(rub_entries))
        except Exception:
            logger.exception("parser.toon_serializer: failed to serialize rubric (continuing)")

    # submission summary + keywords
    lines.append("SUM|SUB:" + _escape_line(submission_summary)[:800])
    if submission_kw:
        lines.append("K|SUB:" + ",".join(submission_kw[:6]))

    # chunks: id:len:s:kw:sum
    ch_lines = []
    for c in chunks:
        cid = c.get("id")
        clen = c.get("len", 0)
        s = round(c.get("rel", 0.0), 2)
        kw = ",".join(c.get("kw", [])[:3])
        ssum = _escape_line(c.get("sum", ""))[:400]
        ch_lines.append(f"{cid}:{clen}:s={s}:k={kw}:sum={ssum}")
    if ch_lines:
        lines.append("CH|" + ";".join(ch_lines))

    # instruction placeholder (keeps prompt clean)
    lines.append("IN|Eval: return JSON {\"priority_chunks\",\"quick_score\",\"flags\"}")

    return "\n".join(lines)

def build_toon_text(doc_id: str, prompt_version: str, meta: Dict[str, Any],
                    submission_summary: str, submission_kw: List[str],
                    chunks: List[Dict[str, Any]], rubric: Optional[Dict[str, Any]] = None) -> str:
    """
    Wrapper: try external python-toon first, then fallback to compact textual form.
    External lib will be passed a normalized/minified JSON object for best compression.
    """
    # Prepare a normalized object for external libs:
    payload = {
        "header": {"id": doc_id, "pv": prompt_version},
        "meta": minify_keys(meta or {}),
        "rubric": rubric or {},
        "submission": {"summary": submission_summary, "keywords": submission_kw},
        "chunks": [{"id": c.get("id"), "len": c.get("len"), "rel": c.get("rel"), "kw": c.get("kw"), "sum": c.get("sum")} for c in chunks],
    }

    # Try external lib
    external = _use_external_toon(payload)
    if external:
        return external

    # Fallback: use compact textual TOONv1 format (string)
    return build_compact_toon_text(doc_id, prompt_version, meta, submission_summary, submission_kw, chunks, rubric)
