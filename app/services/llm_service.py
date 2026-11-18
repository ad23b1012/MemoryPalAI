# app/services/llm_service.py
import os
import time
import logging
from typing import Optional, List
from app.config import GOOGLE_API_KEY, GOOGLE_API_KEY_FALLBACK, GEMINI_MODEL

logger = logging.getLogger(__name__)

# Global client and model caches
_client_cache = {}
_model_cache = {}


def _normalize_model_name(model_name: Optional[str]) -> str:
    name = (model_name or GEMINI_MODEL or "gemini-3.8-flash").strip()
    name = name.replace("models/", "")
    return name or "gemini-3.8-flash"


def get_api_keys() -> List[str]:
    """Returns unique ordered list of configured Google API keys."""
    keys = []
    # 1. Primary key(s) - supports comma-separated list
    primary = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY or "")
    if primary:
        for k in primary.split(","):
            cleaned = k.strip()
            if cleaned and cleaned not in keys:
                keys.append(cleaned)

    # 2. Secondary / Fallback keys
    fallback = os.getenv("GOOGLE_API_KEY_FALLBACK", os.getenv("GOOGLE_API_KEY_2", GOOGLE_API_KEY_FALLBACK or ""))
    if fallback:
        for k in fallback.split(","):
            cleaned = k.strip()
            if cleaned and cleaned not in keys:
                keys.append(cleaned)

    return keys


def get_genai_client(api_key: Optional[str] = None):
    """Initializes or retrieves a cached google.genai Client."""
    target_key = api_key or (get_api_keys()[0] if get_api_keys() else "")
    if not target_key:
        logger.warning("No Google API Key detected.")
        return None

    if target_key in _client_cache:
        return _client_cache[target_key]

    try:
        from google import genai
        client = genai.Client(api_key=target_key)
        _client_cache[target_key] = client
        return client
    except Exception as e:
        logger.error(f"Failed to initialize google.genai Client: {e}")
        return None


class GeminiModelWrapper:
    """
    Lightweight wrapper around google.genai Client providing backwards-compatible
    generate_content() interface for agents with automatic multi-key failover.
    """
    def __init__(self, model_name: str):
        self.model_name = _normalize_model_name(model_name)

    def generate_content(self, contents, **kwargs):
        return generate_with_retry(self, contents)


def get_llm(model_name: Optional[str] = None):
    """
    Returns an initialized Gemini model interface using gemini-3.8-flash.
    """
    selected_model = _normalize_model_name(model_name)
    if selected_model in _model_cache:
        return _model_cache[selected_model]

    wrapper = GeminiModelWrapper(selected_model)
    _model_cache[selected_model] = wrapper
    logger.info(f"Gemini LLM initialized with model: {selected_model} (via modern google.genai SDK)")
    return wrapper


def generate_with_retry(model, prompt: str, retries: int = 2, delay: float = 1.0) -> str:
    """
    Generate text content using gemini-3.8-flash with automatic multi-key failover.
    If Key 1 hits a quota limit (429) or error, automatically switches to Key 2.
    """
    if not prompt or not prompt.strip():
        return ""

    keys = get_api_keys()
    if not keys:
        return (
            "⚠️ **Google API Key not detected.**\n\n"
            "Please configure your `GOOGLE_API_KEY` in `.env` to enable intelligent cognitive answers."
        )

    model_name = getattr(model, "model_name", None) or _normalize_model_name(None)

    # Multi-Key Failover Loop: Try Key 1 first; on 429/error, fail over to Key 2
    for key_idx, current_key in enumerate(keys):
        client = get_genai_client(current_key)
        if client is None:
            continue

        for attempt in range(retries):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if hasattr(resp, "text") and resp.text:
                    return resp.text.strip()
                if hasattr(resp, "candidates") and resp.candidates:
                    cand = resp.candidates[0]
                    if hasattr(cand, "content") and cand.content.parts:
                        parts = [p.text for p in cand.content.parts if hasattr(p, "text") and p.text]
                        if parts:
                            return "\n".join(parts).strip()
                return str(resp).strip()

            except Exception as e:
                err_msg = str(e)
                logger.warning(f"Gemini call with API Key {key_idx + 1} attempt {attempt + 1}/{retries} failed: {err_msg}")

                # Fast break on network unavailability
                if "nodename nor servname" in err_msg.lower() or "errno 8" in err_msg.lower():
                    logger.warning("Network connection unavailable; breaking to pedagogical fallback.")
                    return ""

                # If rate limited (429) or auth error and another key is available, fail over immediately
                if ("429" in err_msg or "quota" in err_msg.lower() or "403" in err_msg or "401" in err_msg) and key_idx < len(keys) - 1:
                    logger.info(f"API Key {key_idx + 1} quota/auth reached. Failing over to Key {key_idx + 2}...")
                    break  # Break inner attempt loop to try next key in outer loop

                # Model 404 fallback
                if "404" in err_msg or "not found" in err_msg.lower():
                    for fb in ["gemini-3.8-pro", "gemini-2.5-flash"]:
                        try:
                            logger.info(f"Retrying prompt with fallback model: {fb}")
                            resp = client.models.generate_content(model=fb, contents=prompt)
                            if hasattr(resp, "text") and resp.text:
                                return resp.text.strip()
                        except Exception:
                            continue

                if attempt < retries - 1:
                    time.sleep(delay)

    return ""
