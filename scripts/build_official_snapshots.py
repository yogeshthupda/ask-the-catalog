"""Render locally downloaded UC public webpage bodies as traceable PDF snapshots.

These are project-created PDFs of official *webpages*, NOT university-issued PDFs.
Their printed page numbers reflect this export only. Always obtain instructor
approval before treating snapshots as an accepted final hackathon document set.
"""
from __future__ import annotations

import html as html_module
import re
from datetime import datetime, timezone
from pathlib import Path

from lxml import html
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/official_web_sources"
OUTPUT_DIR = ROOT / "data/raw_pdfs"

PAGES = {
    "executive_resources": ("Executive Graduate Student Resources", "https://www.ucumberlands.edu/admissions/international-students/executive-graduate-student-resources"),
    "registrar": ("Registrar: Academic Records and Graduation", "https://www.ucumberlands.edu/academics/registrar"),
    "sap": ("Satisfactory Academic Progress Policy", "https://www.ucumberlands.edu/admission-aid/financial-aid-tuition/financial-resources/sap"),
    "complaints": ("Student Complaint Procedure", "https://www.ucumberlands.edu/student-life/student-services/student-complaint-procedure"),
    "accessibility": ("Accommodations and Accessibility", "https://www.ucumberlands.edu/student-life/student-services/health-wellness/accommodations-and-accessibility"),
    "admission": ("Admission Requirements", "https://www.ucumberlands.edu/admissions/admission-requirements"),
}


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def blocks(body):
    for element in body.iter():
        tag = element.tag.lower() if isinstance(element.tag, str) else ""
        if tag not in {"h1", "h2", "h3", "h4", "p", "li", "tr"}:
            continue
        if tag == "p" and any(ancestor.tag in ("li", "tr") for ancestor in element.iterancestors()):
            continue
        if tag == "li" and any(ancestor.tag == "li" for ancestor in element.iterancestors()):
            continue
        if tag == "tr":
            text = " | ".join(normalize(" ".join(cell.itertext())) for cell in element
                               if getattr(cell, "tag", "") in ("td", "th"))
        else:
            text = normalize(" ".join(element.itertext()))
        if not text or len(text) < 3:
            continue
        yield tag, text


def split_long(s: str, max_chars: int = 650):
    """Split long source paragraphs without paraphrasing their actual text."""
    if len(s) <= max_chars:
        yield s
        return
    words = s.split()
    current = []
    for word in words:
        if current and len(" ".join(current + [word])) > max_chars:
            yield " ".join(current)
            current = []
        current.append(word)
    if current:
        yield " ".join(current)


def main() -> None:
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    if (font_dir / "DejaVuSans.ttf").exists():
        pdfmetrics.registerFont(TTFont("UCBody", str(font_dir / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("UCBodyBold", str(font_dir / "DejaVuSans-Bold.ttf")))
        regular, bold = "UCBody", "UCBodyBold"
    else:
        regular, bold = "Helvetica", "Helvetica-Bold"
    normal = ParagraphStyle("body", fontName=regular, fontSize=8.6, leading=13, spaceAfter=6,
                            textColor=colors.HexColor("#17202A"))
    title = ParagraphStyle("title", parent=normal, fontName=bold, fontSize=18, leading=24,
                           spaceAfter=11, textColor=colors.HexColor("#12355B"))
    heading = ParagraphStyle("heading", parent=normal, fontName=bold, fontSize=11.5,
                             leading=16, spaceBefore=10, spaceAfter=5,
                             textColor=colors.HexColor("#114E84"))
    small = ParagraphStyle("small", parent=normal, fontSize=7.7, leading=11,
                           textColor=colors.HexColor("#546477"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for slug, (page_title, url) in PAGES.items():
        source = SOURCE_DIR / f"{slug}.html"
        if not source.exists():
            raise FileNotFoundError(f"Missing downloaded official source HTML: {source}")
        root = html.fromstring(source.read_bytes())
        body = root.xpath('//div[contains(@class, "node__content")]')
        if not body:
            raise ValueError(f"Couldn't isolate official article body in {source}")
        story = [
            Paragraph(html_module.escape(page_title), title),
            Paragraph("OFFICIAL UNIVERSITY WEBPAGE SNAPSHOT (PROJECT-CREATED PDF, NOT AN OFFICIAL CATALOG)", small),
            Paragraph("Publisher: University of the Cumberlands", small),
            Paragraph(f"Original URL: {html_module.escape(url)}", small),
            Paragraph("Captured: October 9, 2026. Page numbers below refer to this project-generated PDF export.", small),
            Spacer(1, 16),
        ]
        count = 0
        for kind, content in blocks(body[0]):
            sty = heading if kind.startswith("h") else normal
            prefix = "• " if kind == "li" else ""
            for section in split_long(prefix + content):
                story.append(Paragraph(html_module.escape(section), sty))
                count += 1
        if count < 6:
            raise ValueError(f"Too little official article content extracted: {source}")
        destination = OUTPUT_DIR / f"UC_Web_Snapshot_{slug}.pdf"
        def footer(canvas, doc, source_name=page_title):
            canvas.saveState()
            canvas.setFont(regular, 7)
            canvas.setFillColor(colors.HexColor("#52657A"))
            canvas.drawString(45, 33, "Project-created snapshot of official UC webpage · verify current policy")
            canvas.drawRightString(565, 33, f"PDF p. {doc.page}")
            canvas.restoreState()
        doc = SimpleDocTemplate(str(destination), pagesize=letter, rightMargin=45,
                                leftMargin=45, topMargin=45, bottomMargin=50,
                                title=f"UC website snapshot: {page_title}")
        doc.build(story, onFirstPage=footer, onLaterPages=footer)
        print(f"{destination.name}: {count} source text blocks")


if __name__ == "__main__":
    main()

