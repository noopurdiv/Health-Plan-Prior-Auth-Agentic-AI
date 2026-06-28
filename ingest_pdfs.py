#!/usr/bin/env python3
"""Run-once script to ingest PDF policy documents into ChromaDB."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.rag.ingest import ingest_pdfs


def main() -> None:
    chunks, files = ingest_pdfs()
    if files == 0:
        print("Warning: No PDFs ingested. Add policy PDFs to knowledge_base/ and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
