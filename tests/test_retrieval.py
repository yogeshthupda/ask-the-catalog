from catalog_rag.models import Chunk
from catalog_rag.retrieval import TfidfRetriever, HybridTfidfRetriever


def corpus():
    return [
        Chunk("a", "policy.pdf", 3, "To transfer credits submit an official transcript and request review."),
        Chunk("b", "policy.pdf", 5, "Library borrowing rules cover books and overdue returns."),
        Chunk("c", "policy.pdf", 9, "Graduation applications require a degree audit."),
    ]


def test_matching_retrieval_and_page():
    result = TfidfRetriever(corpus()).search("How do I transfer credits?", k=2)
    assert result[0].chunk.page == 3
    assert len(result) == 1  # passages with zero similarity are suppressed
    assert result[0].score > 0
    assert "PDF p. 3" in result[0].citation


def test_empty_and_oov_query():
    retriever = TfidfRetriever(corpus())
    assert retriever.search(" ") == []
    assert retriever.search("zxqvbnm") == []


def test_hybrid_offline_retrieval():
    result = HybridTfidfRetriever(corpus()).search("transfered credits application", 5)
    assert result[0].chunk.page == 3

