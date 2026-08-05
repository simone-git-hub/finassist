from pathlib import Path

from finassist.ingest.parse import extract_main_text


def test_extract_main_text_from_fixture() -> None:
    html = Path("tests/fixtures/sample_faq.html").read_text(encoding="utf-8")
    text, title = extract_main_text(html, fallback_title="Fallback title")

    assert "freeze your card" in text.lower()
    assert "mobile app" in text.lower()
    assert title is not None
