# FinAssist RAG Copilot

A retrieval-augmented LLM copilot for finance support-style questions. Built as a portfolio project demonstrating **RAG systems engineering**: ingestion, vector retrieval, guardrails, evaluation, FastAPI deployment, and a Streamlit demo.

## Motivation

Deep learning experience from molecular generative modeling (transformers, diffusion) transfers directly to modern LLM systems. This project applies that discipline to **text-based copilots**: grounded retrieval, refusal when evidence is missing, and measurable quality.

## Architecture

```mermaid
flowchart LR
    Q[User question] --> API[FastAPI /ask]
    API --> G1[Question guardrails]
    G1 --> R[Retriever / Chroma]
    R --> G2[Retrieval threshold]
    G2 --> LLM[LLM backend]
    LLM --> A[Answer + citations]
```

Pipeline stages: **ingest → chunk → embed → index → retrieve → guardrails → generate**.

## Evaluation results (offline corpus, TF-IDF + extractive LLM)

Run: `finassist-eval --config configs/offline.yaml`

| Mode | Recall@5 | Refusal acc. | In-domain answer rate | p50 latency (ms) | p95 latency (ms) |
|------|----------|--------------|------------------------|------------------|------------------|
| Retrieval-only | 1.00 | — | — | — | — |
| RAG | 1.00 | 0.73 | 0.57 | 0 | 1 |
| No-RAG baseline | — | 1.00 | 1.00 | 0 | 0 |

**Ablation (RAG − No-RAG):** retrieval grounding improves citation-backed answers; the no-RAG baseline often returns ungrounded text without refusing (see [docs/DECISIONS.md](./docs/DECISIONS.md)).

**Limitations:** Metrics are on a 2-document offline demo corpus with TF-IDF embeddings and an extractive “LLM”. Semantic embeddings + Ollama/OpenAI improve fluency; expand `data/eval/qa_gold_offline.jsonl` when adding the full public FAQ manifest.

## Quickstart

```bash
cd project/finassist
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# Build corpus + index (offline demo, no downloads)
finassist-ingest --manifest data/raw/manifest.offline.csv
finassist-build-index --config configs/offline.yaml

# Retrieval
finassist-search --config configs/offline.yaml "How do I freeze my card?"

# RAG answers
finassist-ask --config configs/offline.yaml "How do I freeze my card?"

# Evaluation
finassist-eval --config configs/offline.yaml

# API
finassist-api --config configs/offline.yaml
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I freeze my card?"}'

# Streamlit demo (local pipeline mode works without API)
finassist-demo
# Or explicitly:
# streamlit run src/finassist/demo/app.py
```

### Docker

```bash
docker compose up --build
# API:  http://localhost:8000/docs
# Demo: http://localhost:8501
```

## Phase status

| Phase | Status |
|-------|--------|
| 1 Ingestion | Done |
| 2 Index + retrieval | Done |
| 3 RAG + FastAPI | Done |
| 4 Eval + demo + Docker | Done |

For production-quality generation, set `llm.provider: ollama` or `openai` in `configs/default.yaml` and run `finassist-build-index` (semantic embeddings).

## Documentation

| Doc | Description |
|-----|-------------|
| [PLAN.md](./PLAN.md) | Full project plan and CV framing |
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | System design |
| [docs/DECISIONS.md](./docs/DECISIONS.md) | Design tradeoffs |
| [docs/MILESTONES.md](./docs/MILESTONES.md) | Build checklist |

## Tech stack

Python · FastAPI · ChromaDB · sentence-transformers · scikit-learn · Streamlit · Docker

## License

MIT (corpus sources remain subject to their respective terms — see `data/raw/manifest.csv`).
