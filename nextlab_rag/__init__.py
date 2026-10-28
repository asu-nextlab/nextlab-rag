"""Small, persistent Qdrant RAG prototype for the active Next Lab export."""

from .answering import ABSTENTION_TEXT, ask, validate_citations
from .indexing import EMBEDDING_MODEL, index_corpus, search_corpus, split_passages

__all__ = [
    "ABSTENTION_TEXT",
    "EMBEDDING_MODEL",
    "ask",
    "index_corpus",
    "search_corpus",
    "split_passages",
    "validate_citations",
]
