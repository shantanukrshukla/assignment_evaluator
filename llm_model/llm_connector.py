# llm_model/llm_connector.py
import requests
import mlflow
import json
import concurrent.futures
import time
from typing import Optional, Generator
from utils.logging import logger
from utils.config_loader import config

# Ollama configuration (local LLM)
OLLAMA_API_URL = config.get("ollama", {}).get("api_url", "http://localhost:11434/api/generate")
OLLAMA_MODEL_NAME = config.get("ollama", {}).get("model_name", "llama3:8b")

# Hybrid / timeouts
_llm_cfg = (config.get("llm", {}) or {})
HYBRID_CFG = _llm_cfg.get("hybrid", {}) if isinstance(_llm_cfg, dict) else {}
# hedge delay in seconds (ms in config -> convert)
HEDGE_DELAY_S: float = float(HYBRID_CFG.get("hedge_delay_ms", 15000)) / 1000.0
CONNECT_TIMEOUT_S: float = float(HYBRID_CFG.get("connect_timeout_s", 2.0))
READ_TIMEOUT_S: float = float(HYBRID_CFG.get("read_timeout_s", 300.0))

# debug options
_DEBUG_PREVIEW_CHARS = int((config.get("debug", {}) or {}).get("prompt_truncate_chars", 5000))
_DEBUG_ENABLE = bool((config.get("debug", {}) or {}).get("debug_prompts", False))


def _is_ollama_healthy() -> bool:
    try:
        # A simple GET to the base API url; some Ollama installs respond on root
        requests.get(OLLAMA_API_URL, timeout=(CONNECT_TIMEOUT_S, 1.0))
        return True
    except Exception:
        return False


# ------------------------------
# Ollama (non-streaming)
# ------------------------------
@mlflow.trace(
    name="student-ai-llm-call",
    span_type="GENAI",
    attributes={"provider": "ollama", "model": OLLAMA_MODEL_NAME},
)
def ask_ollama(system_prompt: str, user_prompt: str) -> str:
    """
    Blocking call to Ollama. Always returns a string (fallback "[llm_error]" on error).
    """
    if _DEBUG_ENABLE:
        logger.debug("[LLM_DEBUG] ask_ollama system_prompt_len=%d user_prompt_len=%d system_preview=%s",
                     len(system_prompt or ""), len(user_prompt or ""), (system_prompt or "")[:_DEBUG_PREVIEW_CHARS])

    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": f"{system_prompt}\n\nQuestion: {user_prompt}",
        "stream": False
    }
    start = time.time()
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S))
        resp.raise_for_status()
        data = resp.json()

        # Common response keys used by different Ollama/adapter versions
        for key in ("response", "text", "output", "content"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                elapsed = time.time() - start
                try:
                    mlflow.log_metric("llm_latency_seconds", elapsed)
                    mlflow.set_tag("llm_provider", "ollama")
                    mlflow.set_tag("ollama_model", OLLAMA_MODEL_NAME)
                except Exception:
                    logger.debug("mlflow: failed to log latency/tags (non-fatal)")
                return val

        # If server returned a dict with nested structure
        if isinstance(data, dict):
            # attempt to extract first string value
            for v in data.values():
                if isinstance(v, str) and v.strip():
                    elapsed = time.time() - start
                    try:
                        mlflow.log_metric("llm_latency_seconds", elapsed)
                    except Exception:
                        pass
                    return v

        # fallback: return entire body as string if meaningful
        raw_text = resp.text or ""
        if raw_text.strip():
            elapsed = time.time() - start
            try:
                mlflow.log_metric("llm_latency_seconds", elapsed)
            except Exception:
                pass
            return raw_text

        logger.warning("[LLM] Ollama responded but no textual body found; returning error token")
        return "[llm_error:no_text]"
    except requests.Timeout:
        logger.error("[LLM] Timeout contacting Ollama at %s", OLLAMA_API_URL)
    except requests.RequestException as e:
        logger.error("[LLM] RequestException contacting Ollama: %s", e)
    except Exception:
        logger.exception("[LLM] Unexpected error contacting Ollama")
    return "[llm_error]"


# ------------------------------
# Ollama (streaming generator)
# ------------------------------
def ask_ollama_stream(system_prompt: str, user_prompt: str, timeout: float = READ_TIMEOUT_S) -> Generator[str, None, None]:
    """
    Try to stream from Ollama. If streaming fails, fallback to yield the blocking response.
    Yields partial strings (may be full final text if server isn't streaming).
    """
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": f"{system_prompt}\n\nQuestion: {user_prompt}",
        "stream": True
    }
    start = time.time()
    try:
        with requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=(CONNECT_TIMEOUT_S, timeout)) as resp:
            resp.raise_for_status()
            # iterate lines (NDJSON or chunked)
            for chunk in resp.iter_lines(decode_unicode=True):
                if chunk is None:
                    continue
                line = chunk.strip()
                if not line:
                    continue
                # try parsing JSON per-line
                try:
                    obj = json.loads(line)
                    # common keys
                    if isinstance(obj, dict):
                        for key in ("response", "token", "text"):
                            if key in obj and isinstance(obj[key], str):
                                yield obj[key]
                                break
                        else:
                            # no known key: yield raw JSON string
                            yield line
                    else:
                        # not a dict, yield raw line
                        yield line
                except json.JSONDecodeError:
                    # not JSON -> yield raw
                    yield line
            # done streaming
    except Exception as exc:
        logger.exception("[LLM_STREAM] Ollama streaming failed; falling back to non-stream call: %s", exc)
        # fallback: yield the blocking result
        try:
            text = ask_ollama(system_prompt, user_prompt)
            if isinstance(text, str):
                yield text
            else:
                yield str(text or "[llm_error]")
        except Exception:
            logger.exception("[LLM_STREAM] Fallback non-stream also failed")
            yield "[llm_error]"
    finally:
        elapsed = time.time() - start
        try:
            mlflow.log_metric("llm_stream_total_time_seconds", elapsed)
        except Exception:
            pass


# ------------------------------
# Bedrock (non-streaming)
# ------------------------------
@mlflow.trace(
    name="student-ai-llm-call-bedrock",
    span_type="GENAI",
    attributes={"provider": "bedrock", "model": None},
)
def ask_bedrock(system_prompt: str, user_prompt: str) -> str:
    """
    Minimal Bedrock wrapper. Returns a string (empty or '[llm_error]' on failure).
    """
    start = time.time()
    try:
        import boto3  # lazy import
        bedrock_cfg = config.get("bedrock", {})
        region_name = bedrock_cfg.get("region", "us-east-1")
        model_id = bedrock_cfg.get("model_id") or "anthropic.claude-3-5"
        inference_profile_arn = bedrock_cfg.get("inference_profile_arn") or ""
        client = boto3.client("bedrock-runtime", region_name=region_name)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
            "max_tokens": int(bedrock_cfg.get("max_tokens", 1024)),
        }
        if bedrock_cfg.get("temperature") is not None:
            body["temperature"] = float(bedrock_cfg.get("temperature"))

        target_model = inference_profile_arn if inference_profile_arn else model_id
        response = client.invoke_model(
            modelId=target_model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        resp_body = response.get("body")
        if hasattr(resp_body, "read"):
            payload = json.loads(resp_body.read())
        else:
            payload = resp_body if isinstance(resp_body, dict) else {}

        # extract text
        content = payload.get("content") or []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                elapsed = time.time() - start
                try:
                    mlflow.log_metric("llm_latency_seconds", elapsed)
                    mlflow.set_tag("llm_provider", "bedrock")
                    mlflow.set_tag("bedrock_model", target_model)
                except Exception:
                    pass
                return item.get("text", "")
        text = payload.get("text")
        if isinstance(text, str):
            elapsed = time.time() - start
            try:
                mlflow.log_metric("llm_latency_seconds", elapsed)
            except Exception:
                pass
            return text
        return "[llm_error:bedrock_no_text]"
    except Exception:
        logger.exception("[LLM] Bedrock call failed")
        return "[llm_error]"


# ------------------------------
# Unified interfaces (with provider_override support)
# ------------------------------
def _ask_llm_hybrid(system_prompt: str, user_prompt: str) -> str:
    logger.info("Starting hybrid LLM mode (hedge_delay=%.1fs)", HEDGE_DELAY_S)
    # If Ollama isn't healthy, fallback to Bedrock directly
    if not _is_ollama_healthy():
        logger.info("Ollama not healthy; calling Bedrock")
        return ask_bedrock(system_prompt, user_prompt)
    # Default hybrid: attempt ollama non-streaming first, fallback to bedrock
    try:
        return ask_ollama(system_prompt, user_prompt)
    except Exception:
        logger.exception("Hybrid: Ollama primary failed, falling back to Bedrock")
        return ask_bedrock(system_prompt, user_prompt)


def ask_llm(system_prompt: str, user_prompt: str, provider_override: Optional[str] = None) -> str:
    """
    Blocking single-string interface. Always returns a string.

    provider_override: optional string to force provider for this call ("ollama"|"bedrock"|"hybrid").
    """
    cfg_provider = (config.get("llm", {}) or {}).get("provider", "ollama")
    effective_provider = (provider_override or cfg_provider or "ollama").lower()
    logger.info("[LLM] provider=%s model=%s (effective=%s)", cfg_provider, OLLAMA_MODEL_NAME, effective_provider)
    start = time.time()
    try:
        if effective_provider == "bedrock":
            res = ask_bedrock(system_prompt, user_prompt)
        elif effective_provider == "hybrid":
            res = _ask_llm_hybrid(system_prompt, user_prompt)
        else:
            # default: ollama
            res = ask_ollama(system_prompt, user_prompt)
        elapsed = time.time() - start
        try:
            mlflow.log_metric("llm_call_total_seconds", elapsed)
            mlflow.set_tag("configured_provider", cfg_provider)
            mlflow.set_tag("configured_model", OLLAMA_MODEL_NAME if effective_provider != "bedrock" else config.get("bedrock", {}).get("model_id", "bedrock"))
        except Exception:
            logger.debug("mlflow: failed to log llm_call_total_seconds/tags (non-fatal)")
        return res
    except Exception:
        logger.exception("[LLM] ask_llm unexpected error (effective_provider=%s)", effective_provider)
        return "[llm_error]"


def ask_llm_stream(system_prompt: str, user_prompt: str, user_identity: Optional[str] = None,
                   provider_override: Optional[str] = None) -> Generator[str, None, None]:
    """
    Streaming interface. If provider supports streaming yields partial strings.
    On failure always yields safe string.

    provider_override: optional string to force provider for this call ("ollama"|"bedrock"|"hybrid").
    """
    cfg_provider = (config.get("llm", {}) or {}).get("provider", "ollama")
    effective_provider = (provider_override or cfg_provider or "ollama").lower()
    logger.info("[LLM_STREAM] provider=%s user=%s (effective=%s)", cfg_provider, user_identity, effective_provider)
    try:
        try:
            mlflow.set_tag("user", user_identity or "anonymous")
            mlflow.set_tag("configured_provider", cfg_provider)
            mlflow.set_tag("configured_model", OLLAMA_MODEL_NAME if effective_provider != "bedrock" else config.get("bedrock", {}).get("model_id", "bedrock"))
        except Exception:
            logger.exception("[LLM_STREAM] failed to set mlflow tags (non-fatal)")

        if effective_provider == "bedrock":
            # non-streaming; yield final text once
            yield ask_bedrock(system_prompt, user_prompt)
            return
        if effective_provider == "ollama":
            yield from ask_ollama_stream(system_prompt, user_prompt)
            return
        if effective_provider == "hybrid":
            # hybrid streaming implementation: try Ollama stream with hedging to Bedrock
            if not _is_ollama_healthy():
                yield ask_bedrock(system_prompt, user_prompt)
                return

            import threading, queue
            out_q: "queue.Queue[Optional[str]]" = queue.Queue()
            stop_event = threading.Event()

            def run_ollama_stream():
                try:
                    for piece in ask_ollama_stream(system_prompt, user_prompt):
                        out_q.put(piece)
                    out_q.put(None)
                except Exception:
                    logger.exception("Ollama stream thread failed")
                    out_q.put(None)

            t = threading.Thread(target=run_ollama_stream, daemon=True)
            t.start()

            # wait for first piece
            try:
                first_piece = out_q.get(timeout=HEDGE_DELAY_S)
            except Exception:
                first_piece = None

            if first_piece is None:
                # hedged to Bedrock
                logger.info("Hybrid: hedging to Bedrock after %.1fs", HEDGE_DELAY_S)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    bedrock_future = pool.submit(ask_bedrock, system_prompt, user_prompt)
                    try:
                        bedrock_text = bedrock_future.result(timeout=HEDGE_DELAY_S * 5)
                        yield bedrock_text
                        return
                    except Exception:
                        logger.exception("Hybrid: Bedrock hedge failed; draining Ollama output")
                        # drain remaining Ollama
                        while True:
                            try:
                                piece = out_q.get(timeout=1.0)
                                if piece is None:
                                    break
                                yield piece
                            except Exception:
                                break
                        return
            else:
                # Ollama produced first piece
                yield first_piece
                while True:
                    piece = out_q.get()
                    if piece is None:
                        break
                    yield piece
                return

    except Exception:
        logger.exception("[LLM_STREAM] ask_llm_stream unexpected error (effective_provider=%s)", effective_provider)
        yield "[llm_error]"

    # fallback final (shouldn't normally hit)
    yield ask_ollama(system_prompt, user_prompt)
