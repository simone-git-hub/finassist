# FinAssist RAG Copilot

> **Status:** Planning / early development — see [PLAN.md](./PLAN.md) for full build roadmap.

A retrieval-augmented LLM copilot for finance support-style questions. Built as a portfolio project targeting **LLM product engineering** skills (RAG, guardrails, evaluation, API deployment) relevant to neobank / fintech GenAI roles.

## Motivation

Deep learning experience from molecular generative modeling (transformers, diffusion) transfers directly to modern LLM systems. This project applies that engineering discipline to **text-based copilots**: grounded retrieval, refusal when evidence is missing, and measurable quality.

## Planned features

- Public FAQ corpus ingestion + vector index
- RAG answers with source citations
- Guardrails (low-confidence refusal, advice boundaries)
- Evaluation harness (Recall@k, refusal accuracy, latency)
- FastAPI service + Streamlit demo
- Dockerized local deployment

## Quickstart

```bash
# 1. Install
cd project/finassist
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env

# 3. Ingest public FAQ corpus (fetch → parse → chunk)
finassist-ingest
# or smoke test first 2 pages:
finassist-ingest --limit 2

# 4. Build vector index (Phase 2 — not implemented yet)
finassist-build-index

# 5. Run API (Phase 3 — not implemented yet)
uvicorn finassist.api.app:app --reload
```

Processed outputs are written to `data/raw/processed/`:
- `documents.jsonl` — cleaned full documents
- `chunks.jsonl` — chunked text with metadata
- `ingest_summary.json` — run stats

## Documentation

| Doc | Description |
|-----|-------------|
| [PLAN.md](./PLAN.md) | Full 2-week project plan, metrics, CV framing |
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | System design & module boundaries |
| [docs/MILESTONES.md](./docs/MILESTONES.md) | Checkbox tracker |

## Tech stack

Python · FastAPI · ChromaDB · sentence-transformers · Ollama (local LLM) · Streamlit · Docker

## License

MIT (corpus sources remain subject to their respective terms — see `data/raw/manifest.csv`).
