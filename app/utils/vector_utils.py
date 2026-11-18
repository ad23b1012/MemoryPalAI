# app/utils/vector_utils.py
import math
from typing import List, Tuple


def dot_product(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculates dot product between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


def norm(vec: List[float]) -> float:
    """Calculates Euclidean norm (L2) of a vector."""
    if not vec:
        return 0.0
    return math.sqrt(sum(x * x for x in vec))


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Computes cosine similarity between two float vectors.
    Returns a score between -1.0 and 1.0 (typically 0.0 to 1.0 for normalized text vectors).
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product(vec_a, vec_b) / (norm_a * norm_b)


def rank_by_similarity(query_vec: List[float], candidate_vectors: List[Tuple[str, List[float]]], top_k: int = 5):
    """
    Ranks candidates by cosine similarity to query vector.
    """
    scored = [
        (doc_id, cosine_similarity(query_vec, vec))
        for doc_id, vec in candidate_vectors
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
