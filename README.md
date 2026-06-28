# Health Plan Prior Auth Agentic AI

Cotiviti GenAI Developer Intern assessment — **ClearAuth** proof of concept.

## Repository layout

| Item | Description |
|------|-------------|
| [`prior-auth-agent/`](prior-auth-agent/) | Full application code (FastAPI, LangGraph, RAG, dashboard) |
| [`Cotiviti_Assessment_Report.docx`](Cotiviti_Assessment_Report.docx) | Written assessment report |
| [`Cotiviti Assessment Presentation.pptx`](Cotiviti%20Assessment%20Presentation.pptx) | Assessment presentation |

## Quick start

```bash
cd prior-auth-agent
cp .env.example .env   # add ANTHROPIC_API_KEY
pip install -r requirements.txt
python ingest_pdfs.py
.\run_server.ps1
```

Open http://localhost:8000

See [`prior-auth-agent/README.md`](prior-auth-agent/README.md) for full setup, demo script, and architecture.

**Author:** Noopur Shekhar Divekar · June 2026
