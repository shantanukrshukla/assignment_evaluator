# llm_model/system_instructions.py
from typing import List, Dict, Optional
import json


def _format_rubric_for_prompt(rubric: Optional[Dict], max_chars: int = 200000) -> str:
    """
    Robust formatter for rubric content used in system prompts.

    Accepts:
      - dict-like rubric (with either a 'criteria' list or key->value mapping)
      - a plain string (e.g. rubric_logic text)
      - None

    Returns a truncated string representation safe to insert into prompts.
    """
    if not rubric:
        return "Rubric: (none provided) — grade using best judgement."

    # If rubric is a plain string (common when passing rubric_logic), return it (truncated).
    if isinstance(rubric, str):
        r = rubric.strip()
        return r if len(r) <= max_chars else r[:max_chars] + " ...[rubric_truncated]"

    lines = ["Rubric:"]
    # If rubric has a structured 'criteria' list, format each entry
    if isinstance(rubric, dict) and "criteria" in rubric and isinstance(rubric["criteria"], list):
        for c in rubric["criteria"]:
            cid = c.get("id") or c.get("name") or c.get("title") or "<id>"
            title = c.get("title") or ""
            weight = c.get("weight", "n/a")
            desc = c.get("description") or c.get("desc") or ""
            lines.append(f"- {cid}: weight={weight}; {title}. {desc}")
    # Otherwise, if it's a dict-like mapping, print key: value lines
    elif isinstance(rubric, dict):
        for k, v in rubric.items():
            lines.append(f"- {k}: {v}")
    else:
        # Fallback: stringify unknown objects safely
        try:
            s = str(rubric)
            return s if len(s) <= max_chars else s[:max_chars] + " ...[rubric_truncated]"
        except Exception:
            return "Rubric: (unreadable rubric provided)"

    s = "\n".join(lines)
    return s if len(s) <= max_chars else s[:max_chars] + " ...[rubric_truncated]"



def build_system_prompt_from_chunks(compact_chunks: List[Dict],
                                    top_full_texts: List[Dict],
                                    rubric: Optional[Dict] = None,
                                    course_id: Optional[str] = None,
                                    max_context_chars: int = 240000) -> str:
    compact_chunks = compact_chunks or []
    top_full_texts = top_full_texts or []
    rubric_text = _format_rubric_for_prompt(rubric, max_chars=2000)

    header = (
        f"Course: {course_id or 'unknown'}\n"
        f"{rubric_text}\n\n"
        "Rubric precedence (AUTHORITATIVE): The rubric text above contains the authoritative grading steps. "
        "If any other part of this prompt (including examples or schema) appears to conflict with the rubric, "
        "FOLLOW THE RUBRIC. All scoring values (per-criterion and overall) MUST follow the rubric’s own scale.\n\n"
        "Strict Rules (READ CAREFULLY):\n"
        "1) Use ONLY the transcript/course context provided below as the source of truth. Do NOT use any external "
        "knowledge, the internet, training data, or general world knowledge.\n"
        "2) If an assessment or fact **requires** information NOT present in the provided course context, you MUST NOT "
        "hallucinate — instead set 'scores' and 'overall' to null and provide clear feedback explaining what is missing. "
        "Also add 'requires_external_knowledge' (or a similarly descriptive string) to the 'violations' array.\n"
        "3) Do NOT call out or infer content from authors, external references, or any data not explicitly present in the "
        "supplied chunks.\n"
        "4) Return EXACTLY one JSON object and nothing else. Schema: "
        "{\"scores\": {\"<criterion_id>\": <number>, ...} | null, "
        "\"overall\": <number> | null, "
        "\"feedback\": \"string\", "
        "\"violations\": [\"string\", ...]}.\n"
        "Example: {\"scores\": {\"criterion_id\": 5}, \"overall\": 80, \"feedback\": \"Example feedback.\", \"violations\": []}\n\n"
        "Important: You may also return an optional top-level key 'priority_chunks' (array of chunk ids ordered by "
        "importance) to indicate which chunks informed your judgement. Only use chunk ids present in the provided context.\n\n"
    )

    index_lines = []
    for c in compact_chunks:
        cid = c.get("id")
        summary = (c.get("summary") or c.get("text") or "")[:250].replace("\n", " ")
        index_lines.append(f"CHUNK:{cid} => {summary}")
    if not index_lines:
        index_lines = ["NO_CHUNKS_PROVIDED => No course chunks supplied."]

    context_parts = []
    total = 0
    for t in top_full_texts:
        block = f"--- CHUNK:{t.get('id')} ---\n{(t.get('text') or '')}\n"
        if total + len(block) > max_context_chars:
            break
        context_parts.append(block)
        total += len(block)

    # append compact summaries for chunks not in top_full_texts
    for c in compact_chunks:
        if any(t.get("id") == c.get("id") for t in top_full_texts):
            continue
        block = f"--- CHUNK:{c.get('id')} (summary) ---\n{(c.get('summary') or '')}\n"
        if total + len(block) > max_context_chars:
            break
        context_parts.append(block)
        total += len(block)

    context = "\n".join(context_parts)[:max_context_chars]
    system_prompt = header + "### Context index ###\n" + "\n".join(index_lines) + "\n\n### Transcript Context ###\n" + context + "\n### End of Context ###"
    return system_prompt



def build_chunk_system_prompt(chunk_id: int,
                              chunk_text: str,
                              rubric: Optional[Dict] = None,
                              course_id: Optional[str] = None,
                              max_context_chars: int = 100000) -> str:
    rubric_text = _format_rubric_for_prompt(rubric, max_chars=1000)
    block = f"--- CHUNK:{course_id or 'COURSE'}:{chunk_id} ---\n{chunk_text}\n"[:max_context_chars]
    header = (
        f"Course chunk: CHUNK:{course_id or 'COURSE'}:{chunk_id}\n"
        f"{rubric_text}\n\n"
        # Make rubric authoritative at chunk-level as well.
        "IMPORTANT: The rubric above is authoritative for grading this chunk. If any instruction or example elsewhere "
        "in this prompt conflicts with the rubric, FOLLOW THE RUBRIC when assigning scores.\n\n"
        "Rules:\n"
        "1) Evaluate RELATIVE TO THIS CHUNK ONLY. Do NOT use any external knowledge, internet access, or assumptions beyond the text in this chunk.\n"
        "2) If the student submission requires information not in this chunk to be graded, set 'assessments' and 'confidence' to null/0 and explain clearly in 'comments' what is missing; include 'requires_external_knowledge' in 'comments' or as a short tag.\n"
        "3) Return EXACTLY one JSON object: "
        "{\"chunk_id\": int, \"assessments\": {\"<criterion_id\": 0.0-1.0, ...} | null, \"comments\": \"string\", \"confidence\": 0.0-1.0}.\n"
    )
    return header + "\n### Chunk Context ###\n" + block + "\n### End ###"
