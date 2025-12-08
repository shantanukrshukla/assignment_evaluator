# parser/summarizer.py
import re
from typing import List
from collections import Counter

# very lightweight extractive summarizer (deterministic)
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# minimal stopwords set — extend as needed
_STOPWORDS = set([
    "the","and","is","in","to","of","a","for","that","with","on","as","are",
    "it","by","an","be","this","we","or","from","at","which","was","were",
    "has","have","had","but","not","they","their","them","can","could"
])

def _sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

def extractive_summary(text: str, max_sentences: int = 2) -> str:
    """
    Deterministic extractive summary: returns the first N sentences.
    This is intentionally simple and fast (keeps token usage predictable).
    """
    sents = _sentences(text)
    if not sents:
        return ""
    # prefer the first sentence(s) as deterministic baseline
    summary = " ".join(sents[:max_sentences])
    return summary

def top_keywords(text: str, topn: int = 5) -> List[str]:
    """
    Very small frequency-based keyword extractor.
    Returns the topn most frequent alpha words (length >=2), excluding stopwords.
    Deterministic and dependency-free.
    """
    if not text:
        return []
    toks = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    toks = [t for t in toks if t not in _STOPWORDS]
    if not toks:
        return []
    fre = Counter(toks)
    most = [w for w, _ in fre.most_common(topn)]
    return most
