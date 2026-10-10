from catalog_rag.models import Chunk
from catalog_rag.pipeline import CatalogAssistant, REFUSAL
from catalog_rag.retrieval import TfidfRetriever


def assistant(mode="extractive", threshold=0.12):
    chunks = [Chunk("1", "example.pdf", 8, "Official transcript required for transfer credits application.")]
    return CatalogAssistant(TfidfRetriever(chunks), mode=mode, threshold=threshold)


def test_offline_answer_exact_source_and_citation():
    answer = assistant().ask("Is an official transcript required for transfer credits?")
    assert not answer.refused
    assert "Official transcript required" in answer.text
    assert "example.pdf, PDF p. 8" in answer.text


def test_unanswerable_refusal():
    answer = assistant().ask("What is today's cafeteria menu?")
    assert answer.refused
    assert answer.text == REFUSAL


def test_personal_approval_and_live_conditions_are_refused():
    assert assistant().ask("Will my transfer-credit request be approved tomorrow?").refused
    assert assistant().ask("What is the parking occupancy right now?").refused


def test_empty_question():
    assert assistant().ask("  ").refused


def test_ollama_unavailable_falls_back(monkeypatch):
    from catalog_rag import pipeline

    def fail(*args, **kwargs):
        raise OSError("Ollama not installed")

    monkeypatch.setattr(pipeline, "_ollama_generate", fail)
    result = assistant(mode="ollama").ask("What document is needed for transfer credits?")
    assert result.mode == "extractive"
    assert "unavailable" in result.warning

