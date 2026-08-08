from __future__ import annotations

import re
from typing import Protocol

import httpx

from finassist.config import AppConfig, get_settings
from finassist.rag.prompts import build_rag_prompt, format_context_blocks
from finassist.rag.schemas import RefusalReason


class LLMBackend(Protocol):
    def generate(self, *, question: str, context_chunks: list[str], max_words: int) -> str: ...


class ExtractiveLLM:
    """Deterministic fallback that summarizes the top retrieved chunk (no external LLM)."""

    def generate(self, *, question: str, context_chunks: list[str], max_words: int) -> str:
        del question
        if not context_chunks:
            return "I don't have enough information in the available help articles."

        words = context_chunks[0].split()[:max_words]
        summary = " ".join(words)
        return f"Based on the help articles: {summary} [1]"


class MockLLM:
    """Predictable LLM for unit tests."""

    def __init__(self, template: str | None = None) -> None:
        self.template = template or "Answer from context [1]: {context_preview}"

    def generate(self, *, question: str, context_chunks: list[str], max_words: int) -> str:
        del question, max_words
        preview = context_chunks[0][:80] if context_chunks else "no context"
        return self.template.format(context_preview=preview)


class OllamaLLM:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def generate(self, *, question: str, context_chunks: list[str], max_words: int) -> str:
        context = format_context_blocks_from_text(context_chunks)
        prompt = build_rag_prompt(question=question, context=context, max_answer_words=max_words)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Could not reach Ollama at {self.base_url}. "
                "Start Ollama (e.g. run `ollama serve`) and pull the model "
                f"(`ollama pull {self.model}`), or use the offline config: "
                "`finassist-ask --config configs/offline.yaml \"...\"`."
            ) from exc

        return str(data.get("response", "")).strip()


class OpenAILLM:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def generate(self, *, question: str, context_chunks: list[str], max_words: int) -> str:
        context = format_context_blocks_from_text(context_chunks)
        prompt = build_rag_prompt(question=question, context=context, max_answer_words=max_words)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return str(data["choices"][0]["message"]["content"]).strip()


def format_context_blocks_from_text(context_chunks: list[str]) -> str:
    lines: list[str] = []
    for index, text in enumerate(context_chunks, start=1):
        lines.append(f"[{index}] {text}")
    return "\n\n".join(lines)


def create_llm(config: AppConfig) -> LLMBackend:
    settings = get_settings()
    provider = config.llm.provider.lower()

    if provider == "mock":
        return MockLLM()
    if provider == "extractive":
        return ExtractiveLLM()
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when llm.provider=openai")
        return OpenAILLM(
            api_key=settings.openai_api_key,
            model=settings.openai_model or config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )

    return OllamaLLM(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model or config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )
