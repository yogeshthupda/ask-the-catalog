"""Generate a concise, source-honest, single-page technical summary."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


OUT = Path("docs/Group5_Technical_Summary.pdf")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
pdfmetrics.registerFont(TTFont("DejaVu", FONT_PATH))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", FONT_BOLD_PATH))

NAVY = colors.HexColor("#13243B")
TEAL = colors.HexColor("#057A78")
GRAY = colors.HexColor("#475569")
LIGHT = colors.HexColor("#EDF5F4")


def paragraph(c: canvas.Canvas, markup: str, x: float, top: float,
              w: float, size: float = 8.0, leading: float = 11.1,
              color=GRAY) -> float:
    style = ParagraphStyle(
        "text", fontName="DejaVu", fontSize=size, leading=leading,
        textColor=color, alignment=TA_LEFT, spaceAfter=0,
    )
    p = Paragraph(markup, style)
    _, height = p.wrap(w, 600)
    p.drawOn(c, x, top - height)
    return top - height


def heading(c: canvas.Canvas, label: str, x: float, top: float) -> float:
    c.setFillColor(TEAL)
    c.setFont("DejaVu-Bold", 10)
    c.drawString(x, top - 10, label.upper())
    return top - 20


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=letter)
    w, h = letter
    margin = 39
    c.setTitle("Ask the Catalog | Group 5 Technical Summary")
    c.setAuthor("Group 5 / AI-assisted hackathon prototype")
    c.setFillColor(NAVY)
    c.rect(0, h - 110, w, 110, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("DejaVu-Bold", 20)
    c.drawString(margin, h - 43, "ASK THE CATALOG")
    c.setFont("DejaVu", 10)
    c.drawString(margin, h - 64, "Group 5  •  MSDS 630 NLP Hackathon  •  October 9, 2026")
    c.setFillColor(colors.HexColor("#A7E6DA"))
    c.setFont("DejaVu-Bold", 8.2)
    c.drawString(margin, h - 85, "OFFLINE-FIRST PDF RETRIEVAL + GROUNDED EXTRACTIVE ANSWERS")
    y = h - 122

    # Essential source qualification is foregrounded, not buried in footnotes.
    c.setFillColor(LIGHT)
    c.roundRect(margin, y - 53, w - margin * 2, 53, 9, stroke=0, fill=1)
    y = paragraph(c,
        "<b>Provisional source set — not instructor-approved catalog PDFs.</b> "
        "Six project-created snapshots from public official UC webpages; the 2026–27 "
        "catalog host returned HTTP 403 and handbook access was restricted. "
        "Professor approval or a rerun on original catalog/handbook PDFs is still needed.",
        margin + 11, y - 10, w - margin * 2 - 22, size=7.7, leading=10.5, color=NAVY)
    y -= 19

    y = heading(c, "Data and methods", margin, y)
    y = paragraph(c,
        "<b>Sources:</b> executive student resources, registrar, satisfactory academic "
        "progress, student complaints, accommodations, and admission requirements. "
        "Captured October 9, 2026. <b>6 PDFs → 39 extracted pages → 74 page-bound "
        "chunks</b> (350 words with 50-word overlap; PyMuPDF). The UI exposes "
        "one-based PDF page citations and the original source texts.",
        margin, y, w - 2 * margin)
    y -= 8
    y = paragraph(c,
        "<b>Retrieval:</b> baseline TF-IDF word 1–2 grams + cosine; final hybrid "
        "70% word + 30% character 3–5-gram TF-IDF. Top five chunks feed "
        "source-sentence excerpts; citation IDs identify PDFs/pages. Unsupported "
        "and personal/future/live questions trigger refusal. Optional local "
        "embeddings and local Ollama generation are available but <b>not part "
        "of the reported experiment.</b>",
        margin, y, w - 2 * margin)
    y -= 12

    y = heading(c, "Measured development results | 25 answerable + 5 unanswerable", margin, y)
    col_x = [margin, margin + 300, margin + 390, margin + 476]
    c.setFillColor(NAVY)
    c.roundRect(margin, y - 26, w - margin * 2, 26, 5, fill=1, stroke=0)
    for x, label in zip(col_x, ["METRIC", "BASELINE", "CAL. TF-IDF", "HYBRID"]):
        c.setFillColor(colors.white)
        c.setFont("DejaVu-Bold", 7.15)
        c.drawString(x + 7, y - 17, label)
    y -= 28
    for row_i, row in enumerate([
        ("Gold-passage Recall@1", "18/25", "18/25", "16/25"),
        ("Gold-passage Recall@5", "23/25", "23/25", "24/25"),
        ("Correct refusals", "3/5", "5/5", "5/5"),
        ("False refusals", "2/25", "0/25", "0/25"),
        ("Mean reciprocal rank", "0.790", "0.790", "0.755"),
    ]):
        if row_i % 2 == 0:
            c.setFillColor(colors.HexColor("#F1F5F9"))
            c.rect(margin, y - 20, w - margin * 2, 20, fill=1, stroke=0)
        for x, label in zip(col_x, row):
            c.setFillColor(TEAL if x == col_x[-1] else NAVY)
            c.setFont("DejaVu-Bold" if x == col_x[-1] else "DejaVu", 7.6)
            c.drawString(x + 7, y - 13, label)
        y -= 20
    y -= 9
    y = paragraph(c,
        "<b>Passage-level Recall@5</b> requires the exact gold supporting phrase "
        "inside a correctly cited top-five chunk. Separate 8-question calibration "
        "selected thresholds 0.095 / 0.118. The 30 were used during later "
        "iteration: <b>these are not blind held-out results.</b> Seventeen tests "
        "passed. Displayed answer excerpts contain the gold quote in "
        "<b>24/25</b> answerable cases, which is <b>not</b> semantic answer accuracy.",
        margin, y, w - 2 * margin, size=7.7, leading=10.4)
    y -= 11

    y = heading(c, "Observed errors, ethics, and next steps", margin, y)
    y = paragraph(c,
        "<b>Q10:</b> INTR599 supporting chunk was below rank 5. "
        "<b>Q01/Q09:</b> supporting passages were rank 4 and initially omitted "
        "when only three excerpts were shown. <b>Q13/Q15:</b> short sentence "
        "selection omitted Parchment or the second degree deadline; expanded "
        "context fixed visibility. Hybrid improves Recall@5 but lowers Recall@1 "
        "(64% vs 72%). Citation presence is not proof of truthful interpretation; "
        "semantic correctness/faithfulness still require independent human review. "
        "Next: professor-approved corpus, fresh blinded QA set, OCR, table checking, "
        "and reviewer-scored answer/citation quality.",
        margin, y, w - 2 * margin, size=7.65, leading=10.4)
    y -= 12

    y = heading(c, "Five contributions and disclosure", margin, y)
    y = paragraph(c,
        "<b>1 Venkat:</b> integration/README/tests; <b>2 Data:</b> PDFs/provenance/" 
        "chunking; <b>3 Modeling:</b> TF-IDF/hybrid/refusal; <b>4 Evaluation:</b> "
        "30 QA, metrics/errors; <b>5 UI:</b> Streamlit, demo, slides. "
        "ChatGPT/Codex assisted with the implementation, QA drafting, analysis, "
        "report and slides. Student verification, instructor document approval, "
        "and real-device rehearsal remain the group's responsibility.",
        margin, y, w - 2 * margin, size=7.45, leading=10)

    c.setStrokeColor(colors.HexColor("#CAD6DE"))
    c.line(margin, 44, w - margin, 44)
    c.setFillColor(GRAY)
    c.setFont("DejaVu", 6.7)
    c.drawString(margin, 31, "Reproduce: README.md  •  Source URLs: data/SOURCE_PROVENANCE.md")
    c.drawRightString(w - margin, 31, "Group 5  |  Page 1 of 1")
    if y < 53:
        raise RuntimeError(f"Content overflow: bottom at {y:.1f}")
    c.save()
    print(f"Wrote {OUT} (lowest content y={y:.1f})")


if __name__ == "__main__":
    main()
