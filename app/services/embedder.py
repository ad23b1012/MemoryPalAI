# app/services/embedder.py
import logging
import hashlib
import numpy as np
from typing import List, Optional, Any
from app.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION
from app.services.llm_service import get_api_keys

logger = logging.getLogger(__name__)

_client_cache = {}


def _resolve_model_name(model_name: Optional[str]) -> str:
    cleaned = (model_name or EMBEDDING_MODEL or "gemini-embedding-2").strip()
    cleaned = cleaned.replace("models/", "")
    if "text-embedding-004" in cleaned or not cleaned:
        return "gemini-embedding-2"
    return cleaned


def get_genai_client(api_key: Optional[str] = None):
    keys = get_api_keys()
    target_key = api_key or (keys[0] if keys else "")
    if not target_key:
        return None

    if target_key in _client_cache:
        return _client_cache[target_key]

    try:
        from google import genai
        client = genai.Client(api_key=target_key)
        _client_cache[target_key] = client
        return client
    except Exception as e:
        logger.error(f"Failed to initialize google.genai Client for embeddings: {e}")
        return None


def _fallback_pseudo_embedding(text: str, dimension: int = EMBEDDING_DIMENSION) -> List[float]:
    """Deterministic pseudo-embedding when offline or keys unavailable."""
    if not text:
        return [0.0] * dimension

    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dimension).astype(float)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _normalize_vector(vec: List[float], dimension: int = EMBEDDING_DIMENSION) -> List[float]:
    """Ensures unit length and matching dimension."""
    if len(vec) > dimension:
        vec = vec[:dimension]
    elif len(vec) < dimension:
        vec = vec + [0.0] * (dimension - len(vec))

    arr = np.array(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def embed_text(text: str, model_name: Optional[str] = None) -> List[float]:
    """Generates embedding vector (768 dimensions) with multi-key failover."""
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIMENSION

    keys = get_api_keys()
    if not keys:
        return _fallback_pseudo_embedding(text, EMBEDDING_DIMENSION)

    target_model = _resolve_model_name(model_name)

    # Multi-Key Failover
    for key_idx, current_key in enumerate(keys):
        client = get_genai_client(current_key)
        if client is None:
            continue

        try:
            from google.genai import types
            cfg = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION)

            formatted_content = text
            if "gemini-embedding-2" in target_model and not text.startswith("task:"):
                formatted_content = f"task: search result | query: {text}"

            resp = client.models.embed_content(
                model=target_model,
                contents=formatted_content,
                config=cfg
            )

            if hasattr(resp, "embeddings") and resp.embeddings:
                raw_vals = list(resp.embeddings[0].values)
                if "gemini-embedding-2" in target_model and len(raw_vals) == EMBEDDING_DIMENSION:
                    return raw_vals
                return _normalize_vector(raw_vals, EMBEDDING_DIMENSION)

        except Exception as e:
            err_str = str(e).lower()
            if "errno 8" in err_str or "nodename nor servname" in err_str or "dns" in err_str:
                return _fallback_pseudo_embedding(text, EMBEDDING_DIMENSION)

            logger.warning(f"Embedding failed with Key {key_idx + 1}: {e}")
            if key_idx < len(keys) - 1:
                logger.info(f"Failing over embedding to secondary Google API Key (Key {key_idx + 2})...")
                continue

    return _fallback_pseudo_embedding(text, EMBEDDING_DIMENSION)


def embed_documents(texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
    """Batch embedding for multiple texts with multi-key failover."""
    if not texts:
        return []

    keys = get_api_keys()
    if not keys:
        return [_fallback_pseudo_embedding(t, EMBEDDING_DIMENSION) for t in texts]

    target_model = _resolve_model_name(model_name)

    for key_idx, current_key in enumerate(keys):
        client = get_genai_client(current_key)
        if client is None:
            continue

        try:
            from google.genai import types
            cfg = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION)

            batch_size = 20
            all_results = []

            for start in range(0, len(texts), batch_size):
                batch_slice = texts[start:start + batch_size]
                contents = [
                    types.Content(parts=[types.Part.from_text(text=f"title: none | text: {t}")])
                    if "gemini-embedding-2" in target_model
                    else types.Content(parts=[types.Part.from_text(text=t)])
                    for t in batch_slice
                ]

                resp = client.models.embed_content(
                    model=target_model,
                    contents=contents,
                    config=cfg
                )

                if hasattr(resp, "embeddings") and resp.embeddings:
                    for emb in resp.embeddings:
                        vals = list(emb.values)
                        all_results.append(_normalize_vector(vals, EMBEDDING_DIMENSION))
                else:
                    all_results.extend([_fallback_pseudo_embedding(t, EMBEDDING_DIMENSION) for t in batch_slice])

            if len(all_results) == len(texts):
                return all_results

        except Exception as e:
            err_str = str(e).lower()
            if "errno 8" in err_str or "nodename nor servname" in err_str or "dns" in err_str:
                return [_fallback_pseudo_embedding(t, EMBEDDING_DIMENSION) for t in texts]

            logger.warning(f"Batch embedding failed with Key {key_idx + 1}: {e}")
            if key_idx < len(keys) - 1:
                logger.info(f"Failing over batch embedding to Key {key_idx + 2}...")
                continue

    return [embed_text(t, model_name=target_model) for t in texts]


def embed_multimodal(parts: List[Any], model_name: Optional[str] = None) -> List[float]:
    """Generates multimodal embedding for image, audio, video, or PDF using gemini-embedding-2."""
    keys = get_api_keys()
    if not keys:
        return _fallback_pseudo_embedding("multimodal_content", EMBEDDING_DIMENSION)

    target_model = _resolve_model_name(model_name) or "gemini-embedding-2"

    for current_key in keys:
        client = get_genai_client(current_key)
        if client is None:
            continue
        try:
            from google.genai import types
            cfg = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION)
            resp = client.models.embed_content(
                model=target_model,
                contents=parts,
                config=cfg
            )
            if hasattr(resp, "embeddings") and resp.embeddings:
                return _normalize_vector(list(resp.embeddings[0].values), EMBEDDING_DIMENSION)
        except Exception:
            continue

    return _fallback_pseudo_embedding("multimodal_content", EMBEDDING_DIMENSION)
