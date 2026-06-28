import os
import re
from pathlib import Path
from typing import Iterator

import chromadb
import pdfplumber
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "./knowledge_base")
COLLECTION_NAME = "prior_auth_policies"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", "A.", "B.", "C.", ". "]


def _get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def _split_text(text: str) -> list[str]:
    if not text.strip():
        return []

    chunks: list[str] = []
    current = text

    while len(current) > CHUNK_SIZE:
        split_at = -1
        for sep in SEPARATORS:
            idx = current.rfind(sep, 0, CHUNK_SIZE)
            if idx > CHUNK_SIZE // 2:
                split_at = idx + len(sep)
                break

        if split_at == -1:
            split_at = CHUNK_SIZE

        chunk = current[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        current = current[split_at - CHUNK_OVERLAP :].strip() if split_at > CHUNK_OVERLAP else current[split_at:].strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _extract_pdf_text(pdf_path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
    return pages


def _iter_pdf_files(knowledge_base: Path) -> Iterator[Path]:
    for path in sorted(knowledge_base.glob("**/*.pdf")):
        if path.is_file():
            yield path


def ingest_pdfs(
    knowledge_base_path: str | None = None,
    chroma_db_path: str | None = None,
) -> tuple[int, int]:
    kb_path = Path(knowledge_base_path or KNOWLEDGE_BASE_PATH)
    db_path = chroma_db_path or CHROMA_DB_PATH

    if not kb_path.exists():
        kb_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = _get_embedding_function()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    pdf_files = list(_iter_pdf_files(kb_path))
    if not pdf_files:
        print(f"No PDF files found in {kb_path}")
        return 0, 0

    total_chunks = 0
    doc_id = 0

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        source_label = _source_label(pdf_file)
        pages = _extract_pdf_text(pdf_file)

        for page_number, page_text in pages:
            chunks = _split_text(page_text)
            for chunk_index, chunk in enumerate(chunks):
                doc_id += 1
                collection.add(
                    ids=[f"doc_{doc_id}"],
                    documents=[chunk],
                    metadatas=[
                        {
                            "source_file": pdf_file.name,
                            "source": source_label,
                            "page_number": page_number,
                            "chunk_index": chunk_index,
                        }
                    ],
                )
                total_chunks += 1

    count = collection.count()
    print(f"Ingested {total_chunks} chunks from {len(pdf_files)} files into ChromaDB")
    print(f"Collection '{COLLECTION_NAME}' now has {count} documents")
    return total_chunks, len(pdf_files)


def _source_label(pdf_file: Path) -> str:
    name = pdf_file.stem.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()
