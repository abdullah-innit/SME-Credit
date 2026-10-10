"""SME Credit Knowledge Base and Retrieval Package.

Illustrative, grounded knowledge base for policy, eligibility, terms,
score methodology, FAQs, repayment guidance, and responsible lending.
"""

from app.knowledge.retrieval import (
    BM25Retriever,
    KnowledgeMatch,
    KnowledgeRetriever,
    get_knowledge_retriever,
)
from app.knowledge.store import KnowledgeChunk, KnowledgeStore, get_knowledge_store

__all__ = [
    "KnowledgeChunk",
    "KnowledgeStore",
    "get_knowledge_store",
    "BM25Retriever",
    "KnowledgeMatch",
    "KnowledgeRetriever",
    "get_knowledge_retriever",
]
