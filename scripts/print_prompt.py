#!/usr/bin/env python3
"""Small helper to inspect the system_prompt built for a document.
Usage: python scripts/print_prompt.py <doc_id_or_assignment_id>

It reuses the same builders in the codebase and prints concise diagnostics:
- compact_chunks_count
- top_full_texts_count
- system_prompt length and a truncated preview
- compact_chunks preview (first 5)
"""
import sys
from bson import ObjectId
from utils.mongo_client import db
from utils.config_loader import config
from module.worker import _build_compact_chunks, _clean_and_truncate_text, normalize_course_payload  # type: ignore
from llm_model.system_instructions import build_system_prompt_from_chunks

COLL_NAME = config.get("mongo", {}).get("prompt_collection", "prompt_builder")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/print_prompt.py <doc_id_or_assignment_id>")
        sys.exit(2)
    key = sys.argv[1]
    coll = db[COLL_NAME]
    doc = None
    try:
        doc = coll.find_one({"_id": ObjectId(key)})
    except Exception:
        doc = coll.find_one({"assignment_id": key})

    if not doc:
        print("Document not found for", key)
        sys.exit(1)

    # replicate worker logic to build chunks
    raw_chunks = doc.get("course_chunks") or []
    if (not raw_chunks) and doc.get("course_payload"):
        try:
            normalized = normalize_course_payload(doc.get("course_payload") or {})
            rc = []
            for ch in normalized.get("chapters", []):
                for lesson in ch.get("lessons", []):
                    rc.append({
                        "lesson_id": lesson.get("lesson_id"),
                        "text": lesson.get("transcript"),
                        "meta": {"chapter_title": ch.get("chapter_title"), "lesson_title": lesson.get("lesson_title")}
                    })
            if rc:
                raw_chunks = rc
        except Exception as e:
            print("Failed to normalize course_payload:", e)

    submission = doc.get("extracted_excerpt") or ""
    # very small keywords extraction (use worker's top_keywords if needed) - but _build_compact_chunks expects submission_kw list
    # reuse the worker's function by computing submission_kw via simple split
    submission_kw = [w.lower() for w in (submission or "").split() if len(w) > 2][:6]

    compact_chunks = _build_compact_chunks(raw_chunks, submission_kw)
    top_full_texts = []
    # simple selection: take first up to 3
    for i, c in enumerate(compact_chunks[:3]):
        # find raw by id
        idx = c.get("id")
        raw_chunk = raw_chunks[idx] if idx < len(raw_chunks) else None
        if isinstance(raw_chunk, dict):
            ft = _clean_and_truncate_text(raw_chunk.get("text") or raw_chunk.get("content") or "", max_chars=8000)
        else:
            ft = _clean_and_truncate_text(str(raw_chunk), max_chars=8000)
        top_full_texts.append({"id": idx, "text": ft})

    system_prompt = build_system_prompt_from_chunks(
        compact_chunks=compact_chunks,
        top_full_texts=top_full_texts,
        rubric=doc.get("rubric") or {},
        course_id=doc.get("course_id") or doc.get("assignment_id"),
        max_context_chars=int(config.get("llm", {}).get("system_prompt_max_chars", 24000))
    )

    print("doc_id:", str(doc.get("_id")))
    print("assignment_id:", doc.get("assignment_id"))
    print("raw_chunks_count:", len(raw_chunks))
    print("compact_chunks_count:", len(compact_chunks))
    print("top_full_texts_count:", len(top_full_texts))
    print("system_prompt_len:", len(system_prompt))
    print("system_prompt_preview:\n", system_prompt[:4000])

    print("\ncompact_chunks preview (first 5):")
    for c in compact_chunks[:5]:
        print(f"- id={c.get('id')} sim={c.get('similarity')} summary={c.get('summary')[:200]!s}")


if __name__ == '__main__':
    main()

