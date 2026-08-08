from __future__ import annotations

import os

import httpx
import streamlit as st

DEFAULT_API_URL = os.environ.get("FINASSIST_API_URL", "http://127.0.0.1:8000")


def _ask_api(question: str, api_url: str, top_k: int | None) -> dict:
    payload: dict = {"question": question}
    if top_k is not None:
        payload["top_k"] = top_k

    with httpx.Client(timeout=30.0) as client:
        response = client.post(f"{api_url.rstrip('/')}/ask", json=payload)
        response.raise_for_status()
        return response.json()


def _ask_local(question: str, top_k: int | None) -> dict:
    from finassist.config import get_app_config
    from finassist.rag.pipeline import build_pipeline

    pipeline = build_pipeline(get_app_config())
    result = pipeline.ask(question, top_k=top_k)
    return result.model_dump()


def main() -> None:
    st.set_page_config(page_title="FinAssist RAG Copilot", page_icon="💳", layout="wide")
    st.title("FinAssist RAG Copilot")
    st.caption("Finance support copilot demo — retrieval-augmented generation with guardrails")

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Backend", ["Local pipeline", "HTTP API"], index=0)
        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        top_k = st.slider("Top-k retrieval", min_value=1, max_value=10, value=5)

        st.markdown("**Try these**")
        st.markdown("- How do I freeze my card?")
        st.markdown("- How do I dispute a transaction?")
        st.markdown("- Should I invest in crypto?")

    question = st.text_input("Your question", placeholder="How do I freeze my card?")

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Thinking..."):
            try:
                if mode == "HTTP API":
                    payload = _ask_api(question.strip(), api_url, top_k)
                else:
                    payload = _ask_local(question.strip(), top_k)
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                return

        if payload.get("refused"):
            st.warning(f"Refused ({payload.get('refusal_reason')}): {payload.get('answer')}")
        else:
            st.success(payload.get("answer", ""))

        st.metric("Latency (ms)", payload.get("latency_ms", 0))

        sources = payload.get("sources") or []
        if sources:
            st.subheader("Sources")
            for source in sources:
                st.markdown(
                    f"**[{source['index']}] {source['title']}** "
                    f"(score={source['score']:.3f})  \n{source['excerpt']}"
                )


if __name__ == "__main__":
    main()
