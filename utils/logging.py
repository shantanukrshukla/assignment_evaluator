import os
import time
import glob
import zipfile
import threading
import logging
from typing import Optional
from datetime import datetime
from utils.config_loader import config
from logging.handlers import TimedRotatingFileHandler

# ------------------------
# Config
# ------------------------
LOG_DIR = config["logs"]["LOG_DIR"]
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PREFIX = config["logs"]["LOG_FILE_PREFIX"]
GLOBAL_COMPACT_THRESHOLD_BYTES = int(eval(config["logs"]["GLOBAL_COMPACT_THRESHOLD_BYTES"]))
MAINTENANCE_INTERVAL_SECONDS = int(eval(config["logs"]["MAINTENANCE_INTERVAL_SECONDS"]))
ZIP_RETENTION_DAYS = int(config["logs"]["ZIP_RETENTION_DAYS"])


# ------------------------
# Thread-local request context
# ------------------------
_request_context = threading.local()

def set_request_context(client_ip: Optional[str] = None,
                        request_id: Optional[str] = None,
                        http_method: Optional[str] = None,
                        http_path: Optional[str] = None,
                        http_status: Optional[int] = None):
    """
    Set per-request logging context. Call this at request start for basic fields,
    and update http_status after you get the response (middleware example below).
    """
    _request_context.client_ip = client_ip or "-"
    _request_context.request_id = request_id or "-"
    _request_context.http_method = http_method or "-"
    _request_context.http_path = http_path or "-"
    _request_context.http_status = str(http_status) if http_status is not None else "-"

def clear_request_context():
    """Clear per-request context. Call at request end."""
    _request_context.client_ip = None
    _request_context.request_id = None
    _request_context.http_method = None
    _request_context.http_path = None
    _request_context.http_status = None

class RequestContextFilter(logging.Filter):
    """
    Logging filter that injects client_ip, request_id, http_method, http_path, http_status
    into log records. Provides sane defaults when not in a request.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.client_ip = getattr(_request_context, "client_ip", "-")
        record.request_id = getattr(_request_context, "request_id", "-")
        # HTTP fields
        record.http_method = getattr(_request_context, "http_method", "-")
        record.http_path = getattr(_request_context, "http_path", "-")
        record.http_status = getattr(_request_context, "http_status", "-")
        return True

# ------------------------
# Logger setup
# ------------------------
def _make_formatter() -> logging.Formatter:
    fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | %(module)s.%(funcName)s | "
        "thread=%(thread)d | client_ip=%(client_ip)s | req_id=%(request_id)s | "
        "http=%(http_method)s %(http_path)s status=%(http_status)s | %(message)s"
    )
    return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

def _make_daily_file_handler() -> TimedRotatingFileHandler:
    """
    Daily log rotation: new file each day with pattern student_ai_assistant_YYYY-MM-DD.log
    """
    base_filename = os.path.join(LOG_DIR, LOG_FILE_PREFIX + ".log")
    handler = TimedRotatingFileHandler(
        base_filename, when="midnight", interval=1, backupCount=7, encoding="utf-8"
    )
    handler.suffix = "%Y-%m-%d"  # adds date to filename
    handler.setFormatter(_make_formatter())
    handler.addFilter(RequestContextFilter())
    return handler

def setup_logger(name: str = LOG_FILE_PREFIX, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console
        ch = logging.StreamHandler()
        ch.setFormatter(_make_formatter())
        ch.addFilter(RequestContextFilter())
        logger.addHandler(ch)

        # Daily file handler
        fh = _make_daily_file_handler()
        logger.addHandler(fh)

    return logger

# Default module logger
logger = setup_logger()

# ------------------------
# Log Maintenance
# ------------------------
def _iter_log_files() -> list:
    # include daily files and rotated suffixes
    return glob.glob(os.path.join(LOG_DIR, f"{LOG_FILE_PREFIX}*.log*"))

def _iter_zip_files() -> list:
    return glob.glob(os.path.join(LOG_DIR, f"{LOG_FILE_PREFIX}*.zip"))

def _total_size_bytes(paths: list) -> int:
    return sum(os.path.getsize(p) for p in paths if os.path.exists(p))

def _compress_files(files: list, dest_zip: str):
    try:
        with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                arcname = os.path.basename(f)
                try:
                    zf.write(f, arcname)
                except Exception:
                    logger.exception("Failed adding file to zip: %s", f)
        # remove originals after successful archive
        for f in files:
            try:
                os.remove(f)
            except Exception:
                logger.exception("Failed to remove log after compression: %s", f)
    except Exception:
        logger.exception("Compression failed for dest %s", dest_zip)

def _cleanup_old_zips():
    cutoff = time.time() - ZIP_RETENTION_DAYS * 86400
    for z in _iter_zip_files():
        try:
            if os.path.getmtime(z) < cutoff:
                os.remove(z)
                logger.info("Deleted old log archive: %s", z)
        except Exception:
            logger.exception("Failed to delete old archive: %s", z)

def _maintenance_pass():
    try:
        files = _iter_log_files()
        total = _total_size_bytes(files)
        if total > GLOBAL_COMPACT_THRESHOLD_BYTES:
            # compress oldest logs, keep the most recent file(s)
            files.sort(key=os.path.getmtime)
            # pick everything except the latest file
            to_compress = files[:-1]
            if to_compress:
                timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                dest_zip = os.path.join(LOG_DIR, f"{LOG_FILE_PREFIX}_{timestamp}.zip")
                logger.info("Compressing %d log files to %s", len(to_compress), dest_zip)
                _compress_files(to_compress, dest_zip)
        _cleanup_old_zips()
    except Exception:
        logger.exception("Log maintenance failed")

def _maintenance_worker(stop_event: threading.Event):
    while not stop_event.is_set():
        _maintenance_pass()
        stop_event.wait(MAINTENANCE_INTERVAL_SECONDS)

_stop_event = threading.Event()
threading.Thread(target=_maintenance_worker, args=(_stop_event,), daemon=True).start()

def shutdown_logging():
    _stop_event.set()
