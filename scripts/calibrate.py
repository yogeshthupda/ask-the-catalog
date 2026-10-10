"""Calibrate a conservative refusal threshold on separate dev questions.

Never tune this script on the frozen evaluation/questions.csv benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog_rag.evaluation import read_questions
from catalog_rag.pipeline import _needs_unavailable_personal_or_live_data
from catalog_rag.retrieval import create_retriever
from catalog_rag.storage import load_chunks


def calibrate(kind: str, questions_file: str, index: str) -> dict:
    chunks, _ = load_chunks(index)
    searcher = create_retriever(chunks, kind=kind)
    questions = read_questions(questions_file, strict_30=False)
    scored = []
    for q in questions:
        results = searcher.search(q.question)
        score = results[0].score if results else 0.0
        guard = _needs_unavailable_personal_or_live_data(q.question)
        scored.append((q.id, q.answerable, score, guard))
    positives = [r for r in scored if r[1]]
    negatives = [r for r in scored if not r[1]]
    if not positives or not negatives:
        raise ValueError("Calibration requires both answerable and unanswerable questions")

    candidates = [round(n / 1000, 3) for n in range(0, 501)]
    best = None
    for threshold in candidates:
        correct_pos = sum(score >= threshold and not guard for _, _, score, guard in positives)
        correct_neg = sum(score < threshold or guard for _, _, score, guard in negatives)
        sensitivity = correct_pos / len(positives)
        specificity = correct_neg / len(negatives)
        balanced_accuracy = (sensitivity + specificity) / 2
        # Penalize unsupported answers, then choose the LOWEST threshold among ties.
        objective = balanced_accuracy - (1 - specificity) * 0.1
        record = (objective, -threshold, threshold, correct_pos, correct_neg)
        if best is None or record > best:
            best = record
    assert best is not None
    return {
        "retriever": kind,
        "threshold": best[2],
        "dev_correct_answerable": best[3],
        "dev_answerable": len(positives),
        "dev_correct_unanswerable_refusals": best[4],
        "dev_unanswerable": len(negatives),
        "dev_questions": [
            {"id": id_, "answerable": answerable, "score": round(score, 4), "guard": guard}
            for id_, answerable, score, guard in scored
        ],
        "caveat": "Tiny calibration set. This score is not held-out test accuracy."
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retriever", choices=["tfidf", "hybrid"], default="tfidf")
    p.add_argument("--index", default="data/processed/chunks.json")
    p.add_argument("--questions", default="evaluation/calibration_questions.csv")
    p.add_argument("--output", default="evaluation/results/calibration.json")
    args = p.parse_args()
    result = calibrate(args.retriever, args.questions, args.index)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

