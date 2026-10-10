"""Information retrieval over SME Credit knowledge chunks.

Implements native BM25 (Okapi) by default without heavy external downloads,
supports TF-IDF, and provides an optional embeddings provider behind a flag.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Protocol

from app.knowledge.store import KnowledgeChunk, KnowledgeStore, get_knowledge_store


@dataclass
class KnowledgeMatch:
    """A search match from the knowledge base."""

    chunk_id: str
    title: str
    category: str
    score: float
    snippet: str
    citation: str
    content: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "category": self.category,
            "score": round(self.score, 4),
            "snippet": self.snippet,
            "citation": self.citation,
        }


# Tokenizer supporting both English and Urdu / Arabic script
TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)

COMMON_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "is",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "up",
    "about",
    "into",
    "over",
    "after",
    "it",
    "my",
    "your",
    "ka",
    "ki",
    "ke",
    "ko",
    "se",
    "aur",
    "hai",
    "hain",
    "kya",
    "main",
    "par",
    "کا",
    "کی",
    "کے",
    "کو",
    "سے",
    "اور",
    "ہے",
    "ہیں",
    "کیا",
    "میں",
    "پر",
}


def tokenize(text: str) -> list[str]:
    """Tokenize English, Urdu, and alphanumeric words into lowercase tokens."""
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in COMMON_STOPWORDS and len(t) > 1]


class BM25Retriever:
    """Native Okapi BM25 index over KnowledgeChunks."""

    def __init__(self, chunks: list[KnowledgeChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.corpus_size = len(chunks)
        self.doc_tokens: list[list[str]] = []
        self.doc_lens: list[int] = []
        self.avg_dl: float = 0.0
        self.doc_freqs: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Compute document frequencies and IDF table."""
        total_len = 0
        for chunk in self.chunks:
            # Index title, tags, content, and Urdu summary
            full_text = f"{chunk.title} {' '.join(chunk.tags)} {chunk.summary_ur} {chunk.content}"
            tokens = tokenize(full_text)
            self.doc_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_dl = (total_len / self.corpus_size) if self.corpus_size > 0 else 1.0

        for token, freq in self.doc_freqs.items():
            # Standard Lucene/Okapi smoothed IDF
            self.idf[token] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens: list[str]) -> list[float]:
        """Compute BM25 score for each document given tokenized query."""
        scores = [0.0] * self.corpus_size
        if not query_tokens or self.corpus_size == 0:
            return scores

        for token in query_tokens:
            token_idf = self.idf.get(token, 0.0)
            if token_idf <= 0.0:
                continue

            for doc_idx, tokens in enumerate(self.doc_tokens):
                term_count = tokens.count(token)
                if term_count == 0:
                    continue
                doc_len = self.doc_lens[doc_idx]
                numerator = term_count * (self.k1 + 1.0)
                denominator = term_count + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / self.avg_dl)
                )
                scores[doc_idx] += token_idf * (numerator / denominator)

        return scores


class EmbeddingsProvider(Protocol):
    """Protocol for optional vector embeddings provider."""

    def embed_text(self, text: str) -> list[float]: ...

    def embed_corpus(self, texts: list[str]) -> list[list[float]]: ...


class MockVectorEmbeddingsProvider:
    """Lightweight hash-based or bag-of-words vector embedder for testing/offline use."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed_text(self, text: str) -> list[float]:
        tokens = tokenize(text)
        vec = [0.0] * self.dim
        for t in tokens:
            h = hash(t) % self.dim
            vec[h] += 1.0
        # Normalise
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_corpus(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two unit vectors."""
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    return float(dot)


class KnowledgeRetriever:
    """Retrieval service supporting BM25, TF-IDF and optional embeddings."""

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        use_embeddings: bool = False,
        embeddings_provider: EmbeddingsProvider | None = None,
    ) -> None:
        self.store = store or get_knowledge_store()
        self.use_embeddings = use_embeddings
        self.embeddings_provider = embeddings_provider or MockVectorEmbeddingsProvider()
        self.chunks = self.store.list_chunks()
        self.bm25 = BM25Retriever(self.chunks)
        self._doc_embeddings: list[list[float]] | None = None

    def _ensure_doc_embeddings(self) -> None:
        if self._doc_embeddings is None:
            texts = [
                f"{c.title} {' '.join(c.tags)} {c.summary_ur} {c.content}" for c in self.chunks
            ]
            self._doc_embeddings = self.embeddings_provider.embed_corpus(texts)

    def search(
        self,
        query: str,
        top_k: int = 3,
        category: str | None = None,
        min_score: float = 0.05,
    ) -> list[KnowledgeMatch]:
        """Search knowledge base and return top-k matches with citations."""
        if not query.strip() or not self.chunks:
            return []

        query_tokens = tokenize(query)

        # 1. BM25 scores
        bm25_scores = self.bm25.score(query_tokens)

        # 2. If embeddings enabled, blend with vector cosine similarity
        final_scores: list[float] = []
        if self.use_embeddings:
            self._ensure_doc_embeddings()
            q_emb = self.embeddings_provider.embed_text(query)
            for idx, _c in enumerate(self.chunks):
                doc_emb = self._doc_embeddings[idx] if self._doc_embeddings else []
                cos_sim = cosine_similarity(q_emb, doc_emb) if doc_emb else 0.0
                # Normalised blend
                bm25_norm = min(1.0, bm25_scores[idx] / 10.0) if bm25_scores[idx] > 0 else 0.0
                blended = 0.6 * bm25_norm + 0.4 * max(0.0, cos_sim)
                final_scores.append(blended)
        else:
            final_scores = bm25_scores

        # Rank indices
        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda i: final_scores[i],
            reverse=True,
        )

        results: list[KnowledgeMatch] = []
        for idx in ranked_indices:
            score = final_scores[idx]
            chunk = self.chunks[idx]

            # Category filter
            if category and chunk.category.lower() != category.lower():
                continue

            # Fallback if no exact keyword match: allow top-k if min_score is 0 or query has terms
            if score < min_score and len(results) > 0:
                continue

            # Generate snippet (first 180 chars or leading sentences)
            snippet_text = chunk.content.replace("\n", " ").strip()
            snippet = snippet_text[:197] + "..." if len(snippet_text) > 200 else snippet_text

            results.append(
                KnowledgeMatch(
                    chunk_id=chunk.id,
                    title=chunk.title,
                    category=chunk.category,
                    score=score,
                    snippet=snippet,
                    citation=chunk.citation,
                    content=chunk.content,
                )
            )

            if len(results) >= top_k:
                break

        # Fallback: If no match met min_score, return the best candidate
        if not results and ranked_indices:
            best_idx = ranked_indices[0]
            best_chunk = self.chunks[best_idx]
            snippet_text = best_chunk.content.replace("\n", " ").strip()
            snippet = snippet_text[:197] + "..." if len(snippet_text) > 200 else snippet_text
            results.append(
                KnowledgeMatch(
                    chunk_id=best_chunk.id,
                    title=best_chunk.title,
                    category=best_chunk.category,
                    score=final_scores[best_idx],
                    snippet=snippet,
                    citation=best_chunk.citation,
                    content=best_chunk.content,
                )
            )

        return results


# Global default retriever
_retriever_instance: KnowledgeRetriever | None = None


def get_knowledge_retriever(use_embeddings: bool | None = None) -> KnowledgeRetriever:
    """Return singleton knowledge retriever."""
    global _retriever_instance
    from app.core.config import settings

    flag = settings.EMBEDDINGS_ENABLED if use_embeddings is None else use_embeddings
    if _retriever_instance is None or _retriever_instance.use_embeddings != flag:
        _retriever_instance = KnowledgeRetriever(use_embeddings=flag)
    return _retriever_instance
