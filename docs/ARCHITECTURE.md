# Architecture

## Components

### 1. Ingestion (`src/finassist/ingest/`)

**Responsibility:** Turn raw public web pages into normalized text documents.

| Module | Role |
|--------|------|
| `fetch.py` | Download HTML with caching, respect robots.txt |
| `parse.py` | Extract main content (trafilatura or BeautifulSoup) |
| `chunk.py` | Split into overlapping chunks with metadata |

**Chunk metadata (store with each vector):**
```json
{
  "doc_id": "monzo-faq-freeze-card",
  "source_url": "https://...",
  "title": "How to freeze your card",
  "chunk_index": 2,
  "text": "..."
}
```

**Starting hyperparameters:**
- Chunk size: 512 characters (or ~128 tokens)
- Overlap: 64 characters
- Min chunk length: 100 characters (drop noise)

---

### 2. Index (`src/finassist/index/`)

**Responsibility:** Embed chunks and serve similarity search.

| Module | Role |
|--------|------|
| `embedder.py` | Wrap sentence-transformers model |
| `store.py` | Chroma collection CRUD |
| `retriever.py` | `search(query, top_k) -> List[ScoredChunk]` |

**Interface:**
```python
class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]: ...
```

Future: swap Chroma for FAISS or pgvector without changing RAG layer.

---

### 3. RAG (`src/finassist/rag/`)

**Responsibility:** Build grounded prompts and parse LLM output.

| Module | Role |
|--------|------|
| `prompts.py` | System + user templates |
| `generator.py` | LLM call + citation parsing |
| `pipeline.py` | retrieve → guardrail pre-check → generate |

**Flow:**
1. Retrieve top-k chunks
2. Pre-guardrail: if max score < threshold → refuse early (skip LLM)
3. Format context with numbered sources
4. Generate answer
5. Post-guardrail: verify at least one citation referenced

---

### 4. Guardrails (`src/finassist/guardrails/`)

**Responsibility:** Safety and quality gates.

| Check | Trigger | Action |
|-------|---------|--------|
| Low retrieval score | max(similarity) < τ | Refuse |
| No chunks | empty retrieval | Refuse |
| Out-of-domain keywords | optional classifier | Refuse |
| Advice boundary | legal/tax/investment | Template refusal |

Return structured refusal:
```json
{
  "refused": true,
  "reason": "LOW_RETRIEVAL_CONFIDENCE",
  "message": "I don't have enough information..."
}
```

---

### 5. Evaluation (`src/finassist/eval/`)

**Responsibility:** Batch runs on gold Q&A set.

| Metric | Implementation |
|--------|----------------|
| Recall@k | doc_id overlap with gold |
| Refusal accuracy | `should_refuse` vs actual |
| Latency | wall clock per query |
| Citation overlap | token overlap answer ↔ source (heuristic) |

Output: `results/eval_YYYYMMDD.json` + markdown table for README.

---

### 6. API (`src/finassist/api/`)

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| POST | `/ask` | Main RAG query |
| GET | `/metrics` | Optional prometheus-style counters (stretch) |

**Request:**
```json
{ "question": "How do I dispute a transaction?", "top_k": 5 }
```

**Response:**
```json
{
  "answer": "...",
  "sources": [{ "doc_id": "...", "url": "...", "excerpt": "..." }],
  "refused": false,
  "latency_ms": 1240
}
```

---

## Configuration (`configs/default.yaml`)

```yaml
retrieval:
  top_k: 5
  min_score: 0.55
  embedding_model: "BAAI/bge-small-en-v1.5"

chunking:
  size: 512
  overlap: 64

llm:
  provider: "ollama"  # or "openai"
  model: "llama3.2:3b"
  max_tokens: 256
  temperature: 0.1

guardrails:
  refuse_out_of_domain: true
  blocked_patterns:
    - "\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b"
```

---

## Deployment (local)

```text
docker compose up
  ├── api (uvicorn :8000)
  ├── ollama (:11434)   # optional sidecar
  └── streamlit (:8501) # optional
```

Persist Chroma volume at `./data/index/` for fast restarts.
