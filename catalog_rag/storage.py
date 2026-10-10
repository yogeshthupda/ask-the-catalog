"""Portable JSON corpus; the TF-IDF index is rebuilt locally at startup."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import Chunk


def save_chunks(path: str | Path, chunks: list[Chunk], *, metadata: dict | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(path: str | Path) -> tuple[list[Chunk], dict]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("chunks"), list):
        raise ValueError(f"Unsupported chunk index schema: {source}")
    chunks = [Chunk(**item) for item in data["chunks"]]
    if not chunks or any(not c.text.strip() for c in chunks):
        raise ValueError("Chunk index is empty or contains blank chunks")
    return chunks, data.get("metadata", {})

