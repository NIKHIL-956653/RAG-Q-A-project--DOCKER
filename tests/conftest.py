from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "datasets" / "tarnwick-v1"
BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline-offline.toml"


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal, valid PDF with one Helvetica text line per page ('' = blank page)."""
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    page_ids = [4 + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()
    for pid, text in zip(page_ids, pages, strict=True):
        cid = pid + 1
        objects[pid] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {cid} 0 R >>"
        ).encode()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode() if text else b""
        objects[cid] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for oid in sorted(objects):
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode() + objects[oid] + b"\nendobj\n"
    xref = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n".encode() + b"0000000000 65535 f \n"
    for oid in range(1, size):
        out += f"{offsets[oid]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def write_dataset(tmp_path: Path) -> Callable[..., Path]:
    """Create a small dataset directory: write_dataset(docs={path: text}, questions=[...])."""

    def _write(docs: dict[str, str], questions: list[dict[str, Any]], name: str = "tiny") -> Path:
        root = tmp_path / name
        for rel, text in docs.items():
            target = root / "corpus" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        (root / "dataset.toml").write_text(
            f'name = "{name}"\nversion = "0.0.1"\n', encoding="utf-8"
        )
        lines = [json.dumps(q) for q in questions]
        (root / "questions.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return root

    return _write


TINY_DOCS = {
    "alpha.md": "# Alpha\n\nThe alpha widget weighs 12 kg. It is painted red.\n",
    "beta.md": "# Beta\n\nThe beta widget weighs 30 kg. It ships in a wooden crate.\n",
}

TINY_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "q1",
        "category": "factual",
        "answerable": True,
        "question": "How much does the alpha widget weigh?",
        "evidence": [{"doc": "alpha.md", "quote": "The alpha widget weighs 12 kg."}],
        "reference_answer": "12 kg",
        "required_facts": [{"id": "weight", "any_of": ["12 kg"]}],
    },
    {
        "id": "q2",
        "category": "unanswerable",
        "answerable": False,
        "question": "What is the price of the gamma widget?",
        "absent_terms": ["gamma"],
    },
]
