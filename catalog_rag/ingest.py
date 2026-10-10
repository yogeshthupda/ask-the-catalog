"""PDF text extraction and page-preserving chunks. No OCR is performed."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import Chunk, Page


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving human-readable policy wording."""
    text = text.replace("\u00ad", "").replace("\x00", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)  # hyphenated line breaks
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdfs(root: str | Path) -> tuple[list[Page], list[str]]:
    """Return extracted pages and warnings; document IDs are relative paths."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt (PyMuPDF) to ingest PDFs") from exc

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"PDF directory does not exist: {root}")
    paths = sorted(p for p in root.rglob("*.pdf") if p.is_file())
    if not paths:
        raise ValueError(f"No .pdf files found under {root}")

    pages: list[Page] = []
    warnings: list[str] = []
    for path in paths:
        relative_name = path.relative_to(root).as_posix()
        try:
            with fitz.open(path) as pdf:
                if pdf.needs_pass:
                    warnings.append(f"Skipped password-protected PDF: {relative_name}")
                    continue
                for index, pdf_page in enumerate(pdf):
                    text = normalize_text(pdf_page.get_text("text", sort=True))
                    if len(text) < 40:
                        warnings.append(
                            f"Low/empty extraction: {relative_name} PDF page {index + 1}; "
                            "check scanned pages or tables manually."
                        )
                    if text:
                        pages.append(Page(relative_name, index + 1, text))
        except Exception as exc:
            warnings.append(f"Unable to extract {relative_name}: {exc}")
    if not pages:
        raise ValueError("No extractable text found. Check scans, encryption, or PDF paths.")
    return pages, warnings


def chunk_pages(
    pages: list[Page], chunk_words: int = 350, overlap_words: int = 50
) -> list[Chunk]:
    """Split within page boundaries so each citation points to exactly one page."""
    if chunk_words <= 0 or not 0 <= overlap_words < chunk_words:
        raise ValueError("chunk_words must be positive; overlap must be >=0 and smaller")
    chunks: list[Chunk] = []
    for page in pages:
        words = page.text.split()
        if not words:
            continue
        identity = hashlib.sha1(page.document.encode("utf-8")).hexdigest()[:8]
        step = chunk_words - overlap_words
        for part, start in enumerate(range(0, len(words), step), start=1):
            fragment = words[start : start + chunk_words]
            if not fragment:
                break
            chunks.append(
                Chunk(
                    chunk_id=f"{identity}_p{page.page:04d}_c{part:03d}",
                    document=page.document,
                    page=page.page,
                    text=" ".join(fragment),
                )
            )
            if start + chunk_words >= len(words):
                break
    if not chunks:
        raise ValueError("No non-empty text chunks created")
    return chunks

