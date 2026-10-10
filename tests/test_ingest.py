import pytest

from catalog_rag.ingest import chunk_pages, normalize_text, read_pdfs
from catalog_rag.models import Page


def test_chunking_preserves_one_page_and_overlap():
    pages = [Page("source.pdf", 7, " ".join(f"w{i}" for i in range(20)))]
    chunks = chunk_pages(pages, chunk_words=10, overlap_words=2)
    assert len(chunks) == 3
    assert all(c.page == 7 and c.document == "source.pdf" for c in chunks)
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]


def test_chunking_rejects_invalid_settings():
    with pytest.raises(ValueError):
        chunk_pages([Page("a.pdf", 1, "test")], chunk_words=4, overlap_words=4)


def test_text_cleanup():
    assert normalize_text("trans-\nfer\n\n\ncredits") == "transfer\n\ncredits"


def test_extract_page_from_real_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    source = tmp_path / "policies.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((55, 70), "Sample course withdrawal procedures for our synthetic test.")
    pdf.save(source)
    pdf.close()
    pages, warnings = read_pdfs(tmp_path)
    assert len(pages) == 1
    assert pages[0].document == "policies.pdf"
    assert pages[0].page == 1
    assert "withdrawal" in pages[0].text
    assert warnings == []

