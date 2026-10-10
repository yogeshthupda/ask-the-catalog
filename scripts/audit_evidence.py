"""Check whether the exact gold evidence phrase is actually SHOWN to users.

This is an objective display/quote-coverage check, not an answer correctness or
semantic citation faithfulness judgment. Manual review fields remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from catalog_rag.evaluation import read_questions
from catalog_rag.pipeline import CatalogAssistant
from catalog_rag.retrieval import create_retriever
from catalog_rag.storage import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/processed/chunks.json")
    parser.add_argument("--questions", default="evaluation/questions.csv")
    parser.add_argument("--output-prefix", default="evaluation/results/evidence_quote_audit")
    parser.add_argument("--retriever", default="hybrid", choices=["hybrid", "tfidf"])
    parser.add_argument("--threshold", type=float, default=0.118)
    args = parser.parse_args()
    chunks, _ = load_chunks(args.index)
    assistant = CatalogAssistant(create_retriever(chunks, args.retriever), threshold=args.threshold)
    rows = []
    for q in read_questions(args.questions):
        answer = assistant.ask(q.question)
        exact_quote_visible = q.answerable and bool(q.gold_evidence) and (
            " ".join(q.gold_evidence.casefold().split())
            in " ".join(answer.text.casefold().split())
        )
        rows.append({
            "id": q.id,
            "answerable": "yes" if q.answerable else "no",
            "expected_evidence_phrase": q.gold_evidence,
            "verbatim_evidence_visible": "yes" if exact_quote_visible else ("no" if q.answerable else "n/a"),
            "refused": "yes" if answer.refused else "no",
            "note": ("Not a semantic correctness metric; quote appears somewhere in 5 displayed excerpts."
                     if exact_quote_visible else
                     "Gold phrase absent from displayed excerpts; check retrieval."
                     if q.answerable else
                     "Unanswerable label; check refusal."),
        })
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with (prefix.parent / (prefix.name + ".csv")).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    answerable = [row for row in rows if row["answerable"] == "yes"]
    summary = {
        "metric": "verbatim gold evidence phrase visible in user-facing answer",
        "answerable_with_visible_gold_quote": sum(r["verbatim_evidence_visible"] == "yes" for r in answerable),
        "answerable_total": len(answerable),
        "missing_gold_quote_ids": [r["id"] for r in answerable if r["verbatim_evidence_visible"] != "yes"],
        "caution": "Quote visibility does not establish complete answer correctness or citation faithfulness.",
    }
    (prefix.parent / (prefix.name + "_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
