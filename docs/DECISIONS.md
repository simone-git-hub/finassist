# Design decisions

This document records key engineering tradeoffs in FinAssist. It is meant for portfolio reviewers and interview discussions.

---

## 1. ChromaDB + custom embeddings vs. managed vector DB

**Choice:** Local ChromaDB with precomputed `sentence-transformers` / TF-IDF vectors passed at upsert time.

**Why:**
- Zero cloud cost for portfolio/demo usage
- Full control over embedding model swaps and ablations
- Reproducible offline evaluation with the TF-IDF backend

**Tradeoff:** Not production-scale (no sharding, ACLs, or managed HA). A neobank production stack would likely use pgvector, Pinecone, or Weaviate with monitored embedding pipelines.

---

## 2. Extractive LLM fallback vs. requiring Ollama/OpenAI

**Choice:** Default offline config uses `llm.provider: extractive` — a deterministic summarizer over retrieved chunks.

**Why:**
- Reviewers can run the full pipeline without GPU, API keys, or Ollama
- Keeps the portfolio focused on **RAG systems engineering** (retrieval, guardrails, eval) rather than model quality alone

**Tradeoff:** Answers are less fluent than a real LLM. Production would use Ollama/OpenAI with the same retrieval and guardrail shell.

---

## 3. Guardrails before and after retrieval

**Choice:** Multi-stage gates:
1. Question-level refusal (investment advice, obvious out-of-domain)
2. Retrieval confidence threshold
3. Prompt constraint to cite only provided context

**Why:**
- Fintech support must **refuse** more often than generic chatbots
- Cheap deterministic checks reduce latency and hallucination risk before LLM calls

**Tradeoff:** Regex/heuristic guardrails miss edge cases; production would add classifier routing, human review queues, and logged eval loops.

---

## 4. TF-IDF backend for tests; semantic embeddings for demos

**Choice:** Dual embedding backends (`sentence_transformers` and `tfidf`).

**Why:**
- CI/tests stay fast and network-free
- Default demo path can still showcase semantic retrieval with MiniLM/BGE

**Tradeoff:** Metrics are **not comparable** across backends — always report eval config alongside numbers (see README results table).

---

## 5. RAG vs. no-RAG ablation in eval

**Choice:** Built-in eval compares full RAG against `ask_without_rag()` (LLM with empty context).

**Why:**
- Demonstrates retrieval adds value on in-domain questions
- Shows guardrails + retrieval combine for safer refusals on out-of-domain/advice prompts

**Tradeoff:** No-RAG baseline is strawman for fluency; a stronger baseline would be LLM-only with a system prompt but no retrieval (still no grounding).

---

## 6. FastAPI service + optional Streamlit demo

**Choice:** Product-shaped JSON API as primary interface; Streamlit as thin client.

**Why:**
- Mirrors how ML features integrate into real apps
- Streamlit enables quick stakeholder demos without building a frontend

**Tradeoff:** Streamlit is not production UI; mobile/web clients would call `/ask` directly.
