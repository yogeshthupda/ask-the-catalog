"""Auditable benchmark for 25 answerable + 5 unanswerable questions."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .pipeline import CatalogAssistant


FIELDS = ["id", "category", "question", "answerable", "gold_document", "gold_pages", "gold_answer"]


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    category: str
    question: str
    answerable: bool
    gold_document: str
    gold_pages: tuple[int, ...]
    gold_answer: str
    gold_evidence: str = ""


def read_questions(path: str | Path, *, strict_30: bool = True) -> list[EvalQuestion]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(f not in reader.fieldnames for f in FIELDS):
            raise ValueError(f"Question CSV needs these columns: {', '.join(FIELDS)}")
        rows = list(reader)
    errors: list[str] = []
    if strict_30 and len(rows) != 30:
        errors.append(f"Expected exactly 30 questions; got {len(rows)}")
    seen: set[str] = set()
    questions: list[EvalQuestion] = []
    for line, row in enumerate(rows, start=2):
        identifier = (row["id"] or "").strip()
        question = (row["question"] or "").strip()
        answerable_str = (row["answerable"] or "").strip().lower()
        doc = (row["gold_document"] or "").strip()
        pages_str = (row["gold_pages"] or "").strip()
        gold_answer = (row["gold_answer"] or "").strip()
        gold_evidence = (row.get("gold_evidence") or "").strip()
        if not identifier or identifier in seen:
            errors.append(f"Line {line}: missing/duplicate id")
        seen.add(identifier)
        if not question:
            errors.append(f"Line {line}: missing question")
        if answerable_str not in ("yes", "no"):
            errors.append(f"Line {line}: answerable must be yes or no")
        answerable = answerable_str == "yes"
        pages: tuple[int, ...] = ()
        try:
            pages = tuple(int(n.strip()) for n in pages_str.split(";") if n.strip())
            if any(page < 1 for page in pages):
                raise ValueError("page must be positive")
        except ValueError:
            errors.append(f"Line {line}: gold_pages must contain positive PDF page numbers separated by ;")
        if answerable and (not doc or not pages or not gold_answer):
            errors.append(f"Line {line}: answerable item needs gold_document, gold_pages and gold_answer")
        if not answerable and (doc or pages):
            errors.append(f"Line {line}: unanswerable item must not include source/page")
        questions.append(EvalQuestion(identifier, (row["category"] or "").strip(), question,
                                      answerable, doc, pages, gold_answer, gold_evidence))
    if strict_30 and sum(q.answerable for q in questions) != 25:
        errors.append("Evaluation must contain 25 answerable and 5 unanswerable questions")
    if errors:
        raise ValueError("Invalid evaluation set:\n - " + "\n - ".join(errors))
    return questions


def run_evaluation(assistant: CatalogAssistant, questions: list[EvalQuestion]) -> tuple[dict, list[dict]]:
    """Objective retrieval/refusal scores; humans must label correctness/faithfulness."""
    rows: list[dict] = []
    for question in questions:
        result = assistant.ask(question.question, k=5)
        # With gold_evidence, require the actual supporting passage text rather
        # than any unrelated chunk on the same page (stronger than page recall).
        ranks = [i for i, hit in enumerate(result.hits, start=1)
                 if hit.chunk.document.casefold() == question.gold_document.casefold()
                 and hit.chunk.page in question.gold_pages
                 and (not question.gold_evidence or
                      " ".join(question.gold_evidence.casefold().split()) in
                      " ".join(hit.chunk.text.casefold().split()))]
        gold_rank = min(ranks) if ranks else None
        rows.append({
            "id": question.id, "category": question.category, "question": question.question,
            "answerable": "yes" if question.answerable else "no",
            "gold_document": question.gold_document,
            "gold_pages": ";".join(map(str, question.gold_pages)),
            "gold_answer": question.gold_answer,
            "gold_evidence": question.gold_evidence,
            "hit_at_1": int(bool(gold_rank and gold_rank <= 1)) if question.answerable else "",
            "hit_at_5": int(bool(gold_rank and gold_rank <= 5)) if question.answerable else "",
            "gold_rank": gold_rank or "",
            "refused": int(result.refused),
            "top_score": round(result.hits[0].score, 4) if result.hits else 0,
            "predicted_answer": result.text,
            "retrieved_sources": " | ".join(hit.citation for hit in result.hits),
            "answer_correct_manual": "", "citation_faithful_manual": "",
            "review_notes": "",
        })
    answerable = [r for r in rows if r["answerable"] == "yes"]
    unanswerable = [r for r in rows if r["answerable"] == "no"]
    summary = {
        "n_total": len(rows), "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "recall_at_1": sum(r["hit_at_1"] for r in answerable) / len(answerable) if answerable else None,
        "recall_at_5": sum(r["hit_at_5"] for r in answerable) / len(answerable) if answerable else None,
        "mean_reciprocal_rank": (
            sum(1 / r["gold_rank"] if r["gold_rank"] else 0 for r in answerable) / len(answerable)
            if answerable else None
        ),
        "correct_refusals": sum(r["refused"] for r in unanswerable),
        "refusal_accuracy": sum(r["refused"] for r in unanswerable) / len(unanswerable) if unanswerable else None,
        "false_refusals_on_answerable": sum(r["refused"] for r in answerable),
        "manual_review_required": ["answer_correct_manual", "citation_faithful_manual"],
        "gold_match_level": (
            "passage_phrase" if all(q.gold_evidence for q in questions if q.answerable)
            else "document_page_only"
        ),
    }
    return summary, rows


def write_results(output_prefix: str | Path, summary: dict, rows: list[dict]) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.parent / (prefix.name + "_summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (prefix.parent / (prefix.name + "_details.csv")).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

