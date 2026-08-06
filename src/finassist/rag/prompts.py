from __future__ import annotations

from finassist.models import ScoredChunk


def format_context_blocks(chunks: list[ScoredChunk]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(f"[{index}] title={chunk.title}\n{chunk.text}")
    return "\n\n".join(lines)


def build_rag_prompt(*, question: str, context: str, max_answer_words: int) -> str:
    return f"""You are a helpful finance support assistant. Use ONLY the context below.

Rules:
- Cite sources as [1], [2], ... matching the context list.
- If the context does not contain the answer, respond: "I don't have enough information in the available help articles."
- Do not provide legal, tax, or investment advice.
- Be concise (≤ {max_answer_words} words).

Context:
{context}

Question: {question}

Answer:"""
