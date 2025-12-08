# utils/parser.py
from typing import Dict, Any, List, Callable, Optional
from html.parser import HTMLParser
import html
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self._parts: List[str] = []

    def handle_data(self, data: str):
        if data:
            self._parts.append(data)

    def handle_entityref(self, name: str):
        # e.g. &nbsp; &amp;
        self._parts.append(f"&{name};")

    def handle_charref(self, name: str):
        self._parts.append(f"&#{name};")

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = html.unescape(text)
        return " ".join(text.split())

def _fast_html_to_text(html_str: Optional[str]) -> str:
    """Fast HTML -> plain text. Handles None, returns '' for falsy input."""
    if not html_str:
        return ""
    # quick heuristic: if no '<' and no '&', it's probably plain text already
    if "<" not in html_str and "&" not in html_str:
        return " ".join(str(html_str).split())
    s = HTMLStripper()
    try:
        s.feed(html_str)
        s.close()
    except Exception:
        plain = html.unescape(html_str)
        return " ".join(plain.split())
    return s.get_text()

def pick(d: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Return first present non-None value from dict keys, else default."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _process_lesson(lesson: Dict[str, Any], html_to_text_fn: Callable[[Optional[str]], str]) -> Dict[str, Any]:
    lesson_id = pick(lesson, "id", "lesson_id", "lessonId", default="")
    lesson_title = pick(lesson, "title", "name", default="")

    # transcript may be provided as 'transcript' (already plain) or as 'content'/'body' HTML
    transcript_val = lesson.get("transcript")
    if transcript_val is None:
        transcript_val = pick(lesson, "content", "body", "text", default="")

    # If transcript_val has HTML-like content, convert; otherwise normalize whitespace
    if isinstance(transcript_val, str):
        transcript = html_to_text_fn(transcript_val)
    else:
        transcript = str(transcript_val) if transcript_val is not None else ""

    return {
        "lesson_id": str(lesson_id) if lesson_id is not None else "",
        "lesson_title": str(lesson_title) if lesson_title is not None else "",
        "transcript": transcript or ""
    }

def normalize_course_payload(raw_data: Dict[str, Any], *, parallel_threshold: int = 32, max_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Normalize raw payload into:
    {
      "course": {"course_id": str, "course_title": str, "course_description": str},
      "chapters": [
         {"chapter_id": str, "chapter_title": str, "lessons": [
              {"lesson_id": str, "lesson_title": str, "transcript": str}
         ]}
      ]
    }

    - Accepts wrapped payloads {"data": {...}} or raw.
    - Handles 'text_lessons' vs 'lessons', and 'content' (HTML) vs 'transcript'.
    - parallel_threshold: number of lessons across all chapters to trigger threaded processing.
    - max_workers: override threadpool size; default uses min(32, cpu_count*5).
    """
    if isinstance(raw_data, dict) and "data" in raw_data and isinstance(raw_data["data"], dict):
        payload = raw_data["data"]
    else:
        payload = raw_data or {}

    course_id = pick(payload, "id", "course_id", "courseId", default="")
    course_title = pick(payload, "title", "name", "course_title", default="")
    course_description = pick(payload, "description", "course_description", default="")

    course = {
        "course_id": str(course_id) if course_id is not None else "",
        "course_title": str(course_title) if course_title is not None else "",
        "course_description": str(course_description) if course_description is not None else ""
    }

    raw_chapters = payload.get("chapters") or payload.get("sections") or []

    total_lessons = 0
    for ch in raw_chapters:
        if isinstance(ch, dict):
            lessons_raw = ch.get("text_lessons") or ch.get("lessons") or ch.get("textLessons") or []
            total_lessons += len(lessons_raw) if isinstance(lessons_raw, list) else 0

    use_parallel = total_lessons >= parallel_threshold
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) * 5)

    chapters_out: List[Dict[str, Any]] = []
    html_to_text_fn = _fast_html_to_text

    if use_parallel and total_lessons > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            # For each chapter, create tasks for lessons and wait for them
            for ch in raw_chapters:
                if not isinstance(ch, dict):
                    continue
                chap_id = pick(ch, "id", "chapter_id", "chapterId", default="")
                chap_title = pick(ch, "title", "name", default="")

                lessons_raw = ch.get("text_lessons") or ch.get("lessons") or ch.get("textLessons") or []
                if not isinstance(lessons_raw, list):
                    lessons_raw = []

                futures = {ex.submit(_process_lesson, lesson, html_to_text_fn): idx for idx, lesson in enumerate(lessons_raw)}
                lessons_out = [None] * len(lessons_raw)
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        lessons_out[idx] = fut.result()
                    except Exception:
                        # on failure, fallback to minimal structure
                        lessons_out[idx] = {
                            "lesson_id": "",
                            "lesson_title": "",
                            "transcript": ""
                        }

                # filter out any None (shouldn't happen) and keep order
                lessons_out = [l for l in lessons_out if l is not None]
                chapters_out.append({
                    "chapter_id": str(chap_id) if chap_id is not None else "",
                    "chapter_title": str(chap_title) if chap_title is not None else "",
                    "lessons": lessons_out
                })
    else:
        for ch in raw_chapters:
            if not isinstance(ch, dict):
                continue
            chap_id = pick(ch, "id", "chapter_id", "chapterId", default="")
            chap_title = pick(ch, "title", "name", default="")

            lessons_raw = ch.get("text_lessons") or ch.get("lessons") or ch.get("textLessons") or []
            if not isinstance(lessons_raw, list):
                lessons_raw = []

            lessons_out: List[Dict[str, Any]] = []
            for lesson in lessons_raw:
                try:
                    lessons_out.append(_process_lesson(lesson, html_to_text_fn))
                except Exception:
                    lessons_out.append({
                        "lesson_id": "",
                        "lesson_title": "",
                        "transcript": ""
                    })

            chapters_out.append({
                "chapter_id": str(chap_id) if chap_id is not None else "",
                "chapter_title": str(chap_title) if chap_title is not None else "",
                "lessons": lessons_out
            })

    return {"course": course, "chapters": chapters_out}
