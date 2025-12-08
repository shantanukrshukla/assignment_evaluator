# utils/embeddings_transcript.py
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import torch

# Allow override via env var (e.g., "all-MiniLM-L6-v2" or a GPU-enabled model)
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Optional: limit intra-op threads if CPU bound (tune as needed)
# torch.set_num_threads(int(os.getenv("TORCH_THREADS", "4")))

# Load model once per process
_model = None

def _load_model(device: str = None):
    """Internal: load the SentenceTransformer model into global _model."""
    global _model
    if _model is not None:
        return _model
    if device is None:
        # prefer CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # use device string supported by sentence-transformers
    _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model

# Public helper to initialize model for ProcessPoolExecutor initializer
def init_model_for_worker(device: str = None):
    """
    Call this in a worker process once (e.g., initializer of ProcessPoolExecutor).
    """
    _load_model(device=device)


def chunk_text(text: str, chunk_size: int, overlap: int):
    """
    Fast chunker using words. Returns a list of chunks (strings).
    If you want streaming behavior for huge text, convert to generator externally.
    """
    if not text:
        return []
    words = text.split()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    n = len(words)
    while start < n:
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += step
    return chunks


# ---------- Embedding functions ----------
def embed_texts(texts, batch_size: int = 64, show_progress: bool = False, convert_to_list: bool = True):
    """
    Efficiently embed a list of texts in batches.
    Returns either numpy.ndarray (float32) or list(list(float)) if convert_to_list True.

    Usage:
      embeddings = embed_texts(list_of_texts, batch_size=128)
      single = embed_texts([one_text])[0]
    """
    if _model is None:
        _load_model()

    if not texts:
        return [] if convert_to_list else np.zeros((0, _model.get_sentence_embedding_dimension()), dtype=np.float32)

    # SentenceTransformer will handle batching internally, but we explicitly batch to control memory
    all_embs = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        # SentenceTransformers handles device and no_grad internally; this is safe and fast
        embs = _model.encode(
            batch,
            show_progress_bar=False if not show_progress else True,
            convert_to_numpy=True,
            batch_size=len(batch),
            normalize_embeddings=False
        )
        all_embs.append(embs.astype(np.float32, copy=False))

    embs = np.vstack(all_embs)
    if convert_to_list:
        # convert to Python lists (costly) only if you need to serialize to JSON for Redis
        return embs.tolist()
    return embs


def embed_text(text: str):
    """
    Convenience: embed a single string and return list[float] (compatible with your previous code).
    """
    res = embed_texts([text], batch_size=1, convert_to_list=True)
    return res[0] if res else []
