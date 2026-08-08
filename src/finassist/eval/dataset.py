from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from finassist.config import project_root


class GoldExample(BaseModel):
    question: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    category: str = "general"
    should_refuse: bool = False


def load_gold_dataset(path: Path | None = None) -> list[GoldExample]:
    dataset_path = path or project_root() / "data" / "eval" / "qa_gold_offline.jsonl"
    examples: list[GoldExample] = []

    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(GoldExample.model_validate_json(line))

    return examples
