import json
import redis
from typing import List, Dict, Any, Optional, Tuple
from utils.logging import logger
from utils.config_loader import config

# TTL in seconds for lesson data
EXPIRE_SECONDS: int = config["redis"].get("expire_seconds", 0)

# tuned connection pool: adjust max_connections in your config if needed
pool = redis.ConnectionPool(
    host=config["redis"]["host"],
    port=config["redis"]["port"],
    db=config["redis"]["db"],
    max_connections=config.get("redis", {}).get("max_connections", 50),
    decode_responses=True
)
redis_client = redis.Redis(connection_pool=pool)


def _lesson_key(course_id: str, lesson_id: str) -> str:
    """Single list that stores ordered chunk JSON strings for a lesson."""
    return f"course:{course_id}:lesson:{lesson_id}:chunks"


def _course_lessons_key(course_id: str) -> str:
    """Set key that stores lesson ids for a course."""
    return f"course:{course_id}:lessons"


# -------------------------
# Core (existing) API
# -------------------------
def save_chunks(course_id: str, lesson_id: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Save all chunks for a lesson into a Redis LIST (RPUSH), set TTL once and register the lesson in course set.
    This is done with a pipeline to minimize network roundtrips.
    Returns the lesson key (string).
    """
    if not chunks:
        logger.warning("save_chunks called with empty chunks")
        return _lesson_key(course_id, lesson_id)

    key = _lesson_key(course_id, lesson_id)
    course_lessons_key = _course_lessons_key(course_id)

    # serialize once, compact JSON to reduce network size
    values = [json.dumps(c, separators=(",", ":"), ensure_ascii=False) for c in chunks]

    pipe = redis_client.pipeline(transaction=False)
    # Overwrite existing list by deleting and then RPUSH all values
    pipe.delete(key)
    pipe.rpush(key, *values)
    # set TTL for lesson list and the course set
    if EXPIRE_SECONDS:
        pipe.expire(key, int(EXPIRE_SECONDS))
        pipe.expire(course_lessons_key, int(EXPIRE_SECONDS))
    # register lesson id in sort-agnostic set
    pipe.sadd(course_lessons_key, lesson_id)
    pipe.execute()

    logger.info(f"Saved {len(chunks)} chunks to {key}")
    return key


def get_chunks(course_id: str, lesson_id: str) -> List[Dict[str, Any]]:
    """
    Get all chunks for a lesson in order (LRANGE 0 -1). Returns list of dicts.
    """
    key = _lesson_key(course_id, lesson_id)
    raw = redis_client.lrange(key, 0, -1)
    logger.info(f"Fetching {len(raw)} chunks for {course_id}:{lesson_id}")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        try:
            out.append(json.loads(item))
        except Exception:
            logger.exception(f"Failed to decode chunk {i} for {key}, skipping")
    return out


def get_course_chunks(course_id: str, *, sort_lessons: bool = True) -> List[Dict[str, Any]]:
    """
    Get all chunks for the course across all lessons.
    Uses SMEMBERS to fetch lesson IDs, then pipeline LRANGE to fetch each lesson's chunks.
    Returns flattened list of chunks (lesson order preserved if lesson IDs sortable and sort_lessons True).
    """
    course_lessons_key = _course_lessons_key(course_id)
    lesson_ids = list(redis_client.smembers(course_lessons_key) or [])
    logger.info(f"Found {len(lesson_ids)} lesson ids for course {course_id}")

    if not lesson_ids:
        # fallback to legacy keys detection (only if necessary) using SCAN (incremental)
        pattern = f"course:{course_id}:lesson:*:chunks"
        cursor = 0
        legacy_lesson_ids: List[str] = []
        try:
            while True:
                cursor, keys = redis_client.scan(cursor=cursor, match=pattern, count=1000)
                for k in keys:
                    parts = k.split(":")
                    # course:{course_id}:lesson:{lesson_id}:chunks
                    if len(parts) >= 5:
                        legacy_lesson_ids.append(parts[3])
                if int(cursor) == 0:
                    break
            lesson_ids = legacy_lesson_ids
            logger.info(f"Legacy scan found {len(lesson_ids)} lessons for course {course_id}")
        except Exception:
            logger.exception("Legacy keys scan failed; no chunks available")
            return []

    # optionally sort lesson ids numerically when possible (helps preserve structured order)
    if sort_lessons:
        try:
            lesson_ids.sort(key=lambda x: int(x))
        except Exception:
            lesson_ids.sort()  # lexicographic fallback

    # pipeline LRANGE for each lesson
    pipe = redis_client.pipeline(transaction=False)
    keys = [_lesson_key(course_id, lid) for lid in lesson_ids]
    for k in keys:
        pipe.lrange(k, 0, -1)
    results = pipe.execute()

    flat: List[Dict[str, Any]] = []
    total = 0
    for lid, raw_list in zip(lesson_ids, results):
        if not raw_list:
            continue
        for i, item in enumerate(raw_list):
            try:
                flat.append(json.loads(item))
                total += 1
            except Exception:
                logger.exception(f"Failed to decode chunk {i} for lesson {lid}, skipping")
    logger.info(f"Fetched {total} total chunks for course {course_id}")
    return flat


# -------------------------
# New helpers for embedding-based retrieval & session caching
# -------------------------
def get_course_chunk_embeddings(course_id: str) -> List[Tuple[str, int, List[float]]]:
    """
    Fetch only metadata (lesson_id, chunk index, embedding) for all chunks of a course.
    Returns list of (lesson_id, chunk_index, embedding_list).
    This uses a pipelined LRANGE to fetch each lesson's stored chunk JSONs and extracts only embeddings.
    """
    course_lessons_key = _course_lessons_key(course_id)
    lesson_ids = list(redis_client.smembers(course_lessons_key) or [])
    if not lesson_ids:
        return []

    pipe = redis_client.pipeline(transaction=False)
    keys = [_lesson_key(course_id, lid) for lid in lesson_ids]
    for k in keys:
        pipe.lrange(k, 0, -1)
    results = pipe.execute()

    out: List[Tuple[str, int, List[float]]] = []
    for lid, raw_list in zip(lesson_ids, results):
        if not raw_list:
            continue
        for idx, item in enumerate(raw_list):
            if not item:
                continue
            try:
                j = json.loads(item)
                emb = j.get("embedding")
                if emb:
                    # ensure chunk_id presence (fallback to index)
                    cid = j.get("chunk_id", idx)
                    out.append((str(lid), int(cid), emb))
            except Exception:
                logger.exception("Failed to parse chunk metadata (embedding) - skipping")
                continue
    return out


def get_chunks_by_keys(course_id: str, selected: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
    """
    selected: list of (lesson_id, chunk_index)
    Efficiently fetch corresponding chunk JSONs using pipelined LINDEX per lesson.
    Returns flattened list of chunk dicts in the same order as `selected`.
    """
    if not selected:
        return []

    # group lookups by lesson to reduce number of commands
    by_lesson: Dict[str, List[int]] = {}
    for lesson_id, idx in selected:
        by_lesson.setdefault(lesson_id, []).append(idx)

    pipe = redis_client.pipeline(transaction=False)
    lookups: List[Tuple[str, int]] = []
    for lesson_id, indices in by_lesson.items():
        key = _lesson_key(course_id, lesson_id)
        for i in indices:
            pipe.lindex(key, i)
            lookups.append((lesson_id, i))
    results = pipe.execute()

    out: List[Dict[str, Any]] = []
    for (lesson_id, idx), raw in zip(lookups, results):
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except Exception:
            logger.exception(f"Failed to decode chunk {idx} from lesson {lesson_id}, skipping")
            continue
    return out


def cache_session_context(session_id: str, course_id: str, selected: List[Tuple[str, int]], ttl: int = 300) -> None:
    """
    Save selected chunk keys for a session in Redis with a TTL.
    session_id must be unique per user session.
    Value stored as a LIST of "lesson_id:chunk_index" strings.
    """
    if not session_id:
        return
    key = f"session:{session_id}:ctx:{course_id}"
    payload = [f"{lid}:{idx}" for lid, idx in selected]
    pipe = redis_client.pipeline(transaction=False)
    pipe.delete(key)
    if payload:
        pipe.rpush(key, *payload)
        if ttl:
            pipe.expire(key, int(ttl))
    pipe.execute()


def get_cached_session_context(session_id: str, course_id: str) -> List[Tuple[str, int]]:
    """
    Return list of (lesson_id, chunk_index) previously cached for this session/course.
    """
    if not session_id:
        return []
    key = f"session:{session_id}:ctx:{course_id}"
    raw = redis_client.lrange(key, 0, -1)
    out: List[Tuple[str, int]] = []
    for item in raw:
        try:
            lid, idx = item.rsplit(":", 1)
            out.append((lid, int(idx)))
        except Exception:
            continue
    return out
