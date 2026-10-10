import csv
from pathlib import Path

import pytest

from catalog_rag.evaluation import read_questions, run_evaluation, write_results
from catalog_rag.pipeline import CatalogAssistant
from catalog_rag.retrieval import TfidfRetriever
from catalog_rag.storage import load_chunks


ROOT = Path(__file__).resolve().parents[1]


def test_demo_eval_has_30_questions_and_five_unanswerable():
    questions = read_questions(ROOT / "evaluation/demo_questions.csv")
    assert len(questions) == 30
    assert sum(not q.answerable for q in questions) == 5


def test_unfinished_template_must_not_be_scored():
    with pytest.raises(ValueError, match="missing question"):
        read_questions(ROOT / "evaluation/questions_template.csv")


def test_eval_metrics_and_manual_review_columns(tmp_path):
    chunks, metadata = load_chunks(ROOT / "data/demo/demo_chunks.json")
    assert metadata["synthetic_demo"]
    questions = read_questions(ROOT / "evaluation/demo_questions.csv")
    summary, rows = run_evaluation(CatalogAssistant(TfidfRetriever(chunks)), questions)
    assert summary["n_total"] == 30
    assert 0 <= summary["recall_at_5"] <= 1
    assert 0 <= summary["refusal_accuracy"] <= 1
    assert all("citation_faithful_manual" in row for row in rows)
    write_results(tmp_path / "run", summary, rows)
    assert (tmp_path / "run_summary.json").exists()
    with (tmp_path / "run_details.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 30

