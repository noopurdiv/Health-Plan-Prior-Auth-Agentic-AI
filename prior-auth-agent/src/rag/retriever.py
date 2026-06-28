import logging
import os
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# Use cached embedding model without HuggingFace network calls when enabled
if os.getenv("HF_HUB_OFFLINE", "1").lower() in ("1", "true", "yes"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
if os.getenv("TRANSFORMERS_OFFLINE", "1").lower() in ("1", "true", "yes"):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "prior_auth_policies"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_client: Optional[chromadb.PersistentClient] = None
_collection = None
_embedding_fn = None
_initialized = False


def _init_retriever() -> None:
    """Load ChromaDB client and embedding model once (singleton)."""
    global _client, _collection, _embedding_fn, _initialized
    if _initialized:
        return

    logger.info("Initializing retriever singleton (ChromaDB + %s)", EMBEDDING_MODEL)
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")

    _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    _initialized = True
    print("Retriever singleton ready.")


def warmup_retriever() -> None:
    """Pre-warm embedding model and ChromaDB connection at startup."""
    _init_retriever()
    if _collection and _collection.count() > 0:
        retrieve_policy("prior authorization warmup", n_results=1)
        print("Retriever warmup query complete.")


def _get_collection():
    _init_retriever()
    return _collection


def retrieve_policy(query: str, n_results: int = 5) -> List[dict]:
    collection = _get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[query], n_results=min(n_results, collection.count()))

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    items: List[dict] = []
    for doc, meta, distance in zip(documents, metadatas, distances):
        relevance_score = max(0.0, min(1.0, 1.0 - float(distance)))
        source = meta.get("source") or meta.get("source_file", "Unknown Policy")
        items.append(
            {
                "text": doc,
                "source": source,
                "source_file": meta.get("source_file", ""),
                "page_number": meta.get("page_number"),
                "relevance_score": round(relevance_score, 4),
            }
        )

    items.sort(key=lambda x: x["relevance_score"], reverse=True)
    return items


def get_document_count() -> int:
    try:
        collection = _get_collection()
        return collection.count()
    except Exception:
        return 0
