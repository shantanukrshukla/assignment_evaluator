import sys
from pymongo import MongoClient, errors
from utils.config_loader import config
from urllib.parse import quote_plus
from utils.logging import logger

username = quote_plus(config["mongo"].get("username", ""))
password = quote_plus(config["mongo"].get("password", ""))
host = config["mongo"].get("host", "localhost")
port = config["mongo"].get("port", "27017")
db_name = config["mongo"].get("db_name")
auth_source = config["mongo"].get("auth_source", db_name)

if username and password:
    uri = f"mongodb://{username}:{password}@{host}:{port}/{db_name}?authSource={auth_source}"
    logger.info(f"Connecting to MongoDB database: {uri}")
else:
    uri = f"mongodb://{host}:{port}/{db_name}"

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
except errors.OperationFailure as e:
    logger.info("Mongo authentication failed. Please check username/password and authSource.", file=sys.stderr)
    raise
except Exception as e:
    logger.exception("Mongo connection failed:", str(e), file=sys.stderr)
    raise

db = client[db_name]


def _try_int(val):
    try:
        return int(val)
    except Exception:
        return None


def save_raw_lesson(lesson_id: str, raw_data: dict):
    """
    Save raw lesson data only if not already present.
    Robustly checks several possible locations/types for the id,
    and writes a canonical top-level lesson_id on insert.
    """
    collection = db["raw_lessons"]

    # try numeric form if lesson_id looks numeric
    numeric_id = _try_int(lesson_id)
    candidates = []

    # prefer top-level string match
    candidates.append({"lesson_id": lesson_id})
    # top-level numeric if parsed
    if numeric_id is not None:
        candidates.append({"lesson_id": numeric_id})

    # nested common paths (adjust to your payload shape)
    candidates.append({"data.id": lesson_id})
    if numeric_id is not None:
        candidates.append({"data.id": numeric_id})

    candidates.append({"course.course_id": lesson_id})
    if numeric_id is not None:
        candidates.append({"course.course_id": numeric_id})

    try:
        logger.info("Looking up lesson by candidates: %s", candidates)
        existing = None
        for q in candidates:
            existing = collection.find_one(q)
            if existing:
                logger.info("Found existing by %s -> _id=%s", q, existing.get("_id"))
                break

        if existing:
            return {"inserted": False, "lesson_id": lesson_id, "existing_id": str(existing.get("_id"))}

        doc = dict(raw_data)
        canonical_id = numeric_id if numeric_id is not None else lesson_id
        doc["lesson_id"] = canonical_id

        result = collection.insert_one(doc)
        logger.info("Inserted document _id=%s for lesson_id=%s", result.inserted_id, canonical_id)

        inserted_doc = collection.find_one({"_id": result.inserted_id})
        if not inserted_doc:
            logger.error("Insert reported success but doc not found immediately after insert")
            return {"inserted": False, "lesson_id": lesson_id, "error": "not_found_after_insert"}

        return {"inserted": True, "lesson_id": canonical_id, "inserted_id": str(result.inserted_id)}

    except errors.DuplicateKeyError as e:
        logger.warning("DuplicateKeyError inserting lesson_id=%s: %s", lesson_id, e)
        return {"inserted": False, "lesson_id": lesson_id, "error": "duplicate", "details": str(e)}
    except Exception as e:
        logger.exception("Failed to save raw lesson for lesson_id=%s", lesson_id)
        return {"inserted": False, "lesson_id": lesson_id, "error": str(e)}


def save_chunks(lesson_id: str, chunks: list):
    """Save chunks in a dynamic collection, skip duplicates."""
    collection = db[f"lesson_{lesson_id}"]
    for chunk in chunks:
        exists = collection.find_one({
            "lesson_id": chunk["lesson_id"],
            "chunk_id": chunk["chunk_id"]
        })
        if not exists:
            collection.insert_one(chunk)

def get_chunks(lesson_id: str):
    """Retrieve all chunks for a lesson (excluding _id)."""
    collection = db[f"lesson_{lesson_id}"]
    return list(collection.find({}, {"_id": 0}))

# utils/mongo_client.py (add new function)

def save_normalized_course(course_id: str, normalized_data: dict):
    """
    Save normalized course structure if not already present.
    """
    collection = db["raw_courses"]
    existing = collection.find_one({"course.course_id": course_id})
    if not existing:
        collection.insert_one(normalized_data)
        return {"inserted": True, "course_id": course_id}
    return {"inserted": False, "course_id": course_id}
