from __future__ import annotations

import re

import trafilatura
from bs4 import BeautifulSoup


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_main_text(html: str, *, fallback_title: str | None = None) -> tuple[str, str | None]:
    """Extract readable article text from HTML.

    Returns `(text, extracted_title)`.
    """
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        output_format="txt",
    )
    title = trafilatura.extract_metadata(html).title if html else None

    if extracted and len(extracted.strip()) >= 80:
        return _normalize_whitespace(extracted), title or fallback_title

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    article = soup.find("article")
    container = article or soup.find("main") or soup.body
    if container is None:
        text = soup.get_text(separator="\n")
    else:
        text = container.get_text(separator="\n")

    text = _normalize_whitespace(text)
    page_title = title or (soup.title.string.strip() if soup.title and soup.title.string else None)
    return text, page_title or fallback_title
