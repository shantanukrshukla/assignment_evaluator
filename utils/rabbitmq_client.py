# utils/rabbitmq_client.py
"""
Thread-local RabbitMQ helper using pika.BlockingConnection.

Behavior:
 - Each thread gets its own BlockingConnection and channel via threading.local()
 - Publisher and consumer both use per-thread connections (no shared channels)
 - Infrastructure declaration done idempotently; temporary connection used when needed
 - Retry + reconnect on transient failures
 - Consumer uses its own channel and calls start_consuming() safely in that thread
"""

import json
import threading
import time
from typing import Callable, Any, Dict, Optional
import pika
from pika.exceptions import (
    AMQPConnectionError,
    ChannelClosedByBroker,
    AMQPChannelError,
    StreamLostError,
    ConnectionClosed,
    ProbableAuthenticationError,
)
from utils.config_loader import config
from utils.logging import logger

# Config defaults
_rb_cfg = config.get("rabbitmq", {}) or {}
RABBIT_HOST = _rb_cfg.get("host", "localhost")
RABBIT_PORT = int(_rb_cfg.get("port", 5672))
RABBIT_USER = _rb_cfg.get("username", None)
RABBIT_PASS = _rb_cfg.get("password", None)
DEFAULT_INPUT_QUEUE = _rb_cfg.get("input_queue")
DEFAULT_EXCHANGE = _rb_cfg.get("exchange")
DEFAULT_DLX_EXCHANGE = _rb_cfg.get("dlx_exchange")
DEFAULT_DLQ = _rb_cfg.get("dlq_queue", f"{DEFAULT_INPUT_QUEUE}.dlq")

MAX_PUBLISH_RETRIES = int(_rb_cfg.get("publish_max_retries", 3))
MAX_CONSUMER_RETRIES = int(_rb_cfg.get("consumer_max_retries", 5))
RETRY_HEADER = _rb_cfg.get("retry_header", "x-retries")

INITIAL_RECONNECT_DELAY = float(_rb_cfg.get("reconnect_initial_seconds", 1.0))
MAX_RECONNECT_DELAY = float(_rb_cfg.get("reconnect_max_seconds", 30.0))
CONNECT_ATTEMPTS = int(_rb_cfg.get("connection_attempts", 3))
CONNECT_RETRY_DELAY = float(_rb_cfg.get("connection_retry_delay", 2.0))

# Thread-local storage for connection & channel per thread
_thread_local = threading.local()

# A global lock used for operations that must not happen concurrently across threads
# (e.g., eager init that touches management, or optionally a global close_all)
_global_ops_lock = threading.Lock()


def _build_connection_params() -> pika.ConnectionParameters:
    credentials = None
    if RABBIT_USER:
        credentials = pika.PlainCredentials(username=RABBIT_USER, password=RABBIT_PASS or "")
    params = pika.ConnectionParameters(
        host=RABBIT_HOST,
        port=RABBIT_PORT,
        credentials=credentials,
        heartbeat=int(_rb_cfg.get("heartbeat", 60)),
        blocked_connection_timeout=float(_rb_cfg.get("blocked_connection_timeout", 30)),
        connection_attempts=CONNECT_ATTEMPTS,
        retry_delay=CONNECT_RETRY_DELAY,
    )
    return params


def _thread_has_conn() -> bool:
    return getattr(_thread_local, "connection", None) is not None and getattr(_thread_local, "connection", None).is_open


def _ensure_thread_connection():
    """
    Ensure the current thread has an open BlockingConnection in _thread_local.connection.
    Reconnects as needed with exponential backoff.
    """
    if _thread_has_conn():
        return

    delay = INITIAL_RECONNECT_DELAY
    attempts = 0
    params = _build_connection_params()
    while True:
        attempts += 1
        try:
            logger.info("[RABBIT] (thread=%s) Connecting to RabbitMQ (attempt=%d)...", threading.get_ident(), attempts)
            conn = pika.BlockingConnection(params)
            _thread_local.connection = conn
            # create placeholder channel lazily when needed
            _thread_local.channel = None
            logger.info("[RABBIT] (thread=%s) Connection established: %s", threading.get_ident(), conn)
            return
        except ProbableAuthenticationError as exc:
            logger.exception("[RABBIT] (thread=%s) Authentication failed: %s", threading.get_ident(), exc)
            raise
        except Exception as exc:
            logger.warning("[RABBIT] (thread=%s) Connection attempt %d failed: %s. Retrying in %.1fs", threading.get_ident(), attempts, exc, delay)
            try:
                if getattr(_thread_local, "connection", None):
                    try:
                        _thread_local.connection.close()
                    except Exception:
                        pass
                    _thread_local.connection = None
                    _thread_local.channel = None
            except Exception:
                pass
            time.sleep(delay)
            delay = min(delay * 2, MAX_RECONNECT_DELAY)


def _get_thread_channel(create_if_missing: bool = True) -> pika.adapters.blocking_connection.BlockingChannel:
    """
    Return the thread-local channel, creating it if missing (and creating connection if needed).
    Consumer threads should create their own channels (this is fine since channel is per-thread).
    """
    _ensure_thread_connection()
    ch = getattr(_thread_local, "channel", None)
    if ch and getattr(ch, "is_open", False):
        return ch
    if not create_if_missing:
        return None
    # create a channel for this thread
    try:
        ch = _thread_local.connection.channel()
        # try to enable confirms if publisher (best-effort)
        try:
            ch.confirm_delivery()
        except Exception:
            logger.debug("[RABBIT] (thread=%s) confirm_delivery not available", threading.get_ident())
        _thread_local.channel = ch
        logger.debug("[RABBIT] (thread=%s) opened new thread-local channel %s", threading.get_ident(), ch)
        return ch
    except Exception as exc:
        logger.exception("[RABBIT] (thread=%s) Failed to open channel: %s", threading.get_ident(), exc)
        # force recreate connection once and try again
        try:
            try:
                _thread_local.connection.close()
            except Exception:
                pass
        finally:
            _thread_local.connection = None
            _thread_local.channel = None
        _ensure_thread_connection()
        ch = _thread_local.connection.channel()
        _thread_local.channel = ch
        return ch

def _declare_infrastructure(channel: pika.adapters.blocking_connection.BlockingChannel,
                           queue_name: str,
                           exchange_name: str,
                           dlx_exchange: str,
                           dlq_name: str) -> None:
    """
    Robust infra declaration:
      - Declare exchanges (idempotent).
      - Declare DLQ and bind to DLX.
      - Declare main queue directly (durable, with DLX args). DO NOT use passive=True as first attempt.
      - If the provided channel gets closed by broker while declaring, attempt a recovery declare
        using a fresh short-lived channel off the same connection (best-effort).
    """
    try:
        channel.exchange_declare(exchange=exchange_name, exchange_type="fanout", durable=True)
        logger.debug("[RABBIT] Declared/ensured exchange %s", exchange_name)
    except Exception:
        logger.exception("[RABBIT] Failed to declare exchange %s (continuing)", exchange_name)

    try:
        channel.exchange_declare(exchange=dlx_exchange, exchange_type="fanout", durable=True)
        logger.debug("[RABBIT] Declared/ensured DLX exchange %s", dlx_exchange)
    except Exception:
        logger.exception("[RABBIT] Failed to declare DLX exchange %s (continuing)", dlx_exchange)

    # DLQ ensure + bind (idempotent)
    try:
        channel.queue_declare(queue=dlq_name, durable=True)
        try:
            channel.queue_bind(queue=dlq_name, exchange=dlx_exchange)
        except Exception:
            logger.debug("[RABBIT] DLQ bind may already exist or failed; continuing")
        logger.debug("[RABBIT] Ensured DLQ %s bound to %s", dlq_name, dlx_exchange)
    except Exception:
        logger.exception("[RABBIT] Failed to declare/bind DLQ %s (continuing)", dlq_name)

    # Main queue: declare directly (durable) with DLX arg — avoids passive->404->closed channel
    args = {"x-dead-letter-exchange": dlx_exchange}
    try:
        channel.queue_declare(queue=queue_name, durable=True, arguments=args)
        try:
            channel.queue_bind(queue=queue_name, exchange=exchange_name)
        except Exception:
            logger.debug("[RABBIT] bind after declare may have failed; continuing")
        logger.info("[RABBIT] Declared/ensured queue %s bound to %s (DLX=%s)", queue_name, exchange_name, dlx_exchange)
        return
    except ChannelClosedByBroker as e:
        # channel was closed by broker while declaring — attempt recovery on fresh channel
        logger.warning("[RABBIT] Channel closed by broker while declaring queue %s: %s", queue_name, e)
    except Exception as e:
        logger.exception("[RABBIT] Unexpected error declaring queue %s: %s", queue_name, e)

    # Recovery attempt: try a fresh short-lived channel on same connection (best-effort)
    try:
        logger.debug("[RABBIT] Attempting recovery declare for queue %s using fresh channel", queue_name)
        fresh_ch = channel.connection.channel()
        try:
            fresh_ch.queue_declare(queue=queue_name, durable=True, arguments=args)
            try:
                fresh_ch.queue_bind(queue=queue_name, exchange=exchange_name)
            except Exception:
                logger.debug("[RABBIT] bind after fresh declare may have failed; continuing")
            logger.info("[RABBIT] Recovered and declared queue %s", queue_name)
        finally:
            try:
                if fresh_ch and not getattr(fresh_ch, "is_closed", True):
                    fresh_ch.close()
            except Exception:
                pass
    except Exception:
        logger.exception("[RABBIT] Recovery declare for queue %s failed (continuing)", queue_name)

def _serialize_message(message: Dict[str, Any]) -> bytes:
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _publish_raw(body_bytes: bytes,
                 properties: pika.BasicProperties,
                 exchange_name: str,
                 max_retries: int = MAX_PUBLISH_RETRIES) -> None:
    """
    Publish using the current thread's connection/channel. Retries on transient errors.
    """
    attempt = 0
    delay = 1.0
    while attempt < (max_retries + 1):
        attempt += 1
        try:
            ch = _get_thread_channel(create_if_missing=True)
            if ch is None or getattr(ch, "is_closed", True):
                raise AMQPConnectionError("Thread-local channel not available")
            ok = ch.basic_publish(
                exchange=exchange_name,
                routing_key="",
                body=body_bytes,
                properties=properties,
                mandatory=False
            )
            if ok is False:
                raise Exception("Broker did not confirm delivery (basic_publish returned False)")
            logger.debug("[RABBIT] (thread=%s) Published %d bytes to exchange=%s", threading.get_ident(), len(body_bytes), exchange_name)
            return
        except (AMQPConnectionError, ChannelClosedByBroker, AMQPChannelError, StreamLostError, ConnectionClosed) as exc:
            logger.warning("[RABBIT] (thread=%s) Publish attempt %d failed: %s", threading.get_ident(), attempt, exc)
            # reset this thread's connection and channel
            try:
                if getattr(_thread_local, "channel", None):
                    try:
                        _thread_local.channel.close()
                    except Exception:
                        pass
                _thread_local.channel = None
            except Exception:
                pass
            try:
                if getattr(_thread_local, "connection", None):
                    try:
                        _thread_local.connection.close()
                    except Exception:
                        pass
                _thread_local.connection = None
            except Exception:
                pass

            if attempt <= max_retries:
                time.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY)
                continue
            else:
                logger.exception("[RABBIT] (thread=%s) Exhausted publish retries", threading.get_ident())
                raise
        except Exception as exc:
            logger.exception("[RABBIT] (thread=%s) Unexpected publish error (not retrying): %s", threading.get_ident(), exc)
            raise


def publish_message(message: Dict[str, Any],
                    headers: Optional[Dict[str, Any]] = None,
                    exchange_name: Optional[str] = None,
                    queue_name: Optional[str] = None,
                    dlx_exchange: Optional[str] = None,
                    dlq_name: Optional[str] = None) -> None:
    """
    High-level publish:
     - ensures thread-local connection
     - declares infra using a temporary short-lived channel on this thread's connection
     - publishes message via thread-local channel
    """
    exchange_name = exchange_name or DEFAULT_EXCHANGE
    queue_name = queue_name or DEFAULT_INPUT_QUEUE
    dlx_exchange = dlx_exchange or DEFAULT_DLX_EXCHANGE
    dlq_name = dlq_name or DEFAULT_DLQ

    # ensure thread connection exists
    _ensure_thread_connection()

    # Use a short-lived channel on this thread to declare infra, so we don't affect long-lived consumer+publish channels
    temp_ch = None
    try:
        temp_ch = _thread_local.connection.channel()
        try:
            _declare_infrastructure(temp_ch, queue_name, exchange_name, dlx_exchange, dlq_name)
        finally:
            try:
                if temp_ch and not getattr(temp_ch, "is_closed", True):
                    temp_ch.close()
            except Exception:
                pass
    except Exception:
        logger.exception("[RABBIT] (thread=%s) Failed to declare infrastructure for exchange='%s' queue='%s' (continuing)", threading.get_ident(), exchange_name, queue_name)

    props = pika.BasicProperties(
        delivery_mode=2,
        content_type="application/json",
        headers=headers or {}
    )
    body = _serialize_message(message)
    _publish_raw(body, props, exchange_name)


def _republish_for_retry(original_body: bytes,
                         original_props: pika.BasicProperties,
                         exchange_name: str,
                         dlx_exchange: str,
                         dlq_name: str,
                         retry_count: int) -> None:
    headers = dict(original_props.headers or {})
    headers[RETRY_HEADER] = retry_count
    props = pika.BasicProperties(
        delivery_mode=2,
        content_type=original_props.content_type or "application/json",
        headers=headers
    )
    if retry_count > MAX_CONSUMER_RETRIES:
        # move to DLQ using a temporary per-thread channel
        _ensure_thread_connection()
        tmp_ch = None
        try:
            tmp_ch = _thread_local.connection.channel()
            tmp_ch.basic_publish(exchange=dlx_exchange, routing_key="", body=original_body, properties=props)
            logger.warning("[RABBIT] (thread=%s) Moved message to DLQ after %d retries", threading.get_ident(), retry_count)
        except Exception:
            logger.exception("[RABBIT] Failed to move message to DLQ")
        finally:
            try:
                if tmp_ch and not getattr(tmp_ch, "is_closed", True):
                    tmp_ch.close()
            except Exception:
                pass
    else:
        _publish_raw(original_body, props, exchange_name)


def start_consumer(process_fn: Callable[[Dict[str, Any]], None],
                   prefetch_count: int = 4,
                   queue: Optional[str] = None,
                   exchange: Optional[str] = None,
                   dlx_exchange: Optional[str] = None,
                   dlq_name: Optional[str] = None,
                   auto_ack: bool = False) -> None:
    """
    Blocking consumer loop. Run in the thread that will own this consumer.
    Each consumer call creates and uses its own per-thread connection and channel.
    """
    queue_name = queue or DEFAULT_INPUT_QUEUE
    exchange_name = exchange or DEFAULT_EXCHANGE
    dlx_exchange = dlx_exchange or DEFAULT_DLX_EXCHANGE
    dlq_name = dlq_name or DEFAULT_DLQ

    while True:
        try:
            # ensure this thread's connection + channel
            _ensure_thread_connection()
            try:
                consumer_ch = _get_thread_channel(create_if_missing=True)
            except Exception as e:
                logger.exception("[RABBIT] (thread=%s) Failed to create consumer channel: %s", threading.get_ident(), e)
                time.sleep(1)
                continue

            # declare infra idempotently on consumer channel
            try:
                _declare_infrastructure(consumer_ch, queue_name, exchange_name, dlx_exchange, dlq_name)
            except Exception:
                logger.exception("[RABBIT] (thread=%s) Failed to declare infra for consumer (continuing)", threading.get_ident())

            consumer_ch.basic_qos(prefetch_count=prefetch_count)

            def _on_message(ch, method, properties, body):
                delivery_tag = method.delivery_tag
                try:
                    payload = json.loads(body.decode()) if isinstance(body, (bytes, bytearray)) else json.loads(body)
                except Exception:
                    logger.exception("[RABBIT] Failed to decode incoming message; acking to drop")
                    try:
                        ch.basic_ack(delivery_tag=delivery_tag)
                    except Exception:
                        pass
                    return

                try:
                    retries = int((properties.headers or {}).get(RETRY_HEADER, 0))
                except Exception:
                    retries = 0

                try:
                    process_fn(payload)
                    ch.basic_ack(delivery_tag=delivery_tag)
                except Exception as e:
                    logger.exception("[RABBIT] Processing failed (attempt %d): %s", retries + 1, e)
                    try:
                        ch.basic_ack(delivery_tag=delivery_tag)
                    except Exception:
                        try:
                            ch.basic_nack(delivery_tag=delivery_tag, requeue=False)
                        except Exception:
                            pass
                    try:
                        _republish_for_retry(body if isinstance(body, (bytes, bytearray)) else body.encode(), properties, exchange_name, dlx_exchange, dlq_name, retries + 1)
                    except Exception:
                        logger.exception("[RABBIT] Failed to republish failed message; attempting to move to DLQ")
                        try:
                            _republish_for_retry(body if isinstance(body, (bytes, bytearray)) else body.encode(), properties, exchange_name, dlx_exchange, dlq_name, MAX_CONSUMER_RETRIES + 1)
                        except Exception:
                            logger.exception("[RABBIT] Final attempt to move message to DLQ failed")

            consumer_ch.basic_consume(queue=queue_name, on_message_callback=_on_message, auto_ack=auto_ack)
            logger.info("[RABBIT] Started consuming queue=%s exchange=%s prefetch=%d (thread=%s)", queue_name, exchange_name, prefetch_count, threading.get_ident())
            try:
                consumer_ch.start_consuming()
            except Exception as e:
                logger.exception("[RABBIT] Consumer loop terminated unexpectedly: %s", e)
                # close only this thread's channel and connection and retry
                try:
                    if getattr(_thread_local, "channel", None) and not getattr(_thread_local, "channel", "closed") == "closed":
                        try:
                            _thread_local.channel.close()
                        except Exception:
                            pass
                        _thread_local.channel = None
                except Exception:
                    pass
                try:
                    if getattr(_thread_local, "connection", None) and not getattr(_thread_local, "connection", "closed") == "closed":
                        try:
                            _thread_local.connection.close()
                        except Exception:
                            pass
                        _thread_local.connection = None
                except Exception:
                    pass
                time.sleep(1)
                continue

        except AMQPConnectionError as e:
            logger.warning("[RABBIT] AMQP connection error in consumer loop: %s", e)
            time.sleep(2)
            continue
        except Exception as e:
            logger.exception("[RABBIT] Unexpected error in consumer loop: %s", e)
            time.sleep(2)
            continue


def init_rabbitmq(eager: bool = False) -> None:
    """Optionally call at app startup to validate connection and declare infra."""
    if not eager:
        return
    with _global_ops_lock:
        try:
            _ensure_thread_connection()
            # use a temporary channel on this thread to declare infra
            tmp = _thread_local.connection.channel()
            try:
                _declare_infrastructure(tmp, DEFAULT_INPUT_QUEUE, DEFAULT_EXCHANGE, DEFAULT_DLX_EXCHANGE, DEFAULT_DLQ)
            finally:
                try:
                    if tmp and not getattr(tmp, "is_closed", True):
                        tmp.close()
                except Exception:
                    pass
            logger.info("[RABBIT] Eager RabbitMQ initialization done (thread=%s)", threading.get_ident())
        except Exception:
            logger.exception("[RABBIT] Eager init failed (continuing)")


def close() -> None:
    """Close this thread's connection & channel (call from thread that opened them)."""
    try:
        if getattr(_thread_local, "channel", None) and not getattr(_thread_local.channel, "is_closed", True):
            try:
                _thread_local.channel.close()
            except Exception:
                pass
        _thread_local.channel = None
    except Exception:
        pass
    try:
        if getattr(_thread_local, "connection", None) and not getattr(_thread_local.connection, "is_closed", True):
            try:
                _thread_local.connection.close()
            except Exception:
                pass
        _thread_local.connection = None
    except Exception:
        pass
    logger.info("[RABBIT] Closed thread-local connection/channel (thread=%s)", threading.get_ident())


def close_all() -> None:
    """
    Best-effort: close this thread's resources; other threads should call close() themselves.
    There's no reliable way to force-close sockets in other threads from here safely in Python.
    """
    close()
