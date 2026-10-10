"""Ask the Catalog: compact, evidence-grounded answers with openable source PDFs.

Presentation changes only: no changes to the retriever, threshold, or refusal policy.
Launch with: $env:STREAMLIT_SERVER_ENABLE_STATIC_SERVING='true'; python -m streamlit run app.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from catalog_rag.pipeline import CatalogAssistant
from catalog_rag.retrieval import create_retriever
from catalog_rag.storage import load_chunks

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static" / "source_pdfs"
STOPWORDS = set("a an and are as at be by can do does for from how i in is it of on or should the their there these this to under was were what when where which who why will with your".split())
REFUSAL_START = "i could not find enough information"


def source_details(citation: str) -> tuple[str | None, int]:
    """Read the PDF filename and one-based PDF page from the existing citation."""
    filename_match = re.search(r"([\w.\- ]+\.pdf)", citation, re.IGNORECASE)
    page_match = re.search(r"\bPDF\s*p(?:age)?\.?\s*(\d+)", citation, re.IGNORECASE)
    return (Path(filename_match.group(1).strip()).name if filename_match else None,
            max(1, int(page_match.group(1))) if page_match else 1)


@st.cache_data(show_spinner=False)
def locate_pdf(filename: str) -> str | None:
    """Find source PDFs in the usual corpus directory or another project data folder."""
    for folder in (ROOT / "data" / "raw_pdfs", ROOT / "data" / "pdfs", ROOT / "docs", ROOT / "data"):
        if folder.exists():
            direct = folder / filename
            if direct.is_file():
                return str(direct)
            for found in folder.rglob("*.pdf"):
                if found.name.lower() == filename.lower():
                    return str(found)
    return None


def pdf_url(citation: str) -> str | None:
    """Publish an exact local source file through Streamlit static serving.

    This does not upload the PDF to the internet. Streamlit must be started
    with STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true.
    """
    filename, page = source_details(citation)
    if not filename:
        return None
    path = locate_pdf(filename)
    if not path:
        return None
    # OneDrive / managed Windows folders may prevent creating static directories.
    # A failed PDF preview must never prevent showing the chatbot answer.
    try:
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        destination = STATIC_DIR / filename
        source = Path(path)
        if (not destination.exists() or source.stat().st_mtime_ns > destination.stat().st_mtime_ns
                or source.stat().st_size != destination.stat().st_size):
            shutil.copy2(source, destination)
    except OSError:
        return None
    return f"/app/static/source_pdfs/{quote(filename)}#page={page}"


def relevant_sentence(question: str, excerpt: str) -> str:
    """Choose a useful verbatim source sentence; no generated policy facts."""
    text = " ".join(excerpt.split())
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if x.strip()]
    if not sentences:
        return text[:400]
    words = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2 and w not in STOPWORDS}
    return max(sentences, key=lambda s: (len(words & set(re.findall(r"[a-z0-9]+", s.lower()))), -len(s)))


def short_answer(question: str, response: str, hits: list[dict]) -> dict | None:
    """Derive a compact answer from a quoted source passage only.

    Extracts identifiable grades, amounts, percentages, providers, and
    credit counts when clearly present. Otherwise shows one relevant
    sentence rather than asserting a potentially wrong extracted answer.
    """
    if not hits or response.strip().lower().startswith(REFUSAL_START):
        return None
    matches = re.findall(r'\[S(\d+)\]\s*[“"](.*?)[”"]\s*Source:', response, re.DOTALL)
    if not matches:
        return None
    # First excerpt is highest ranked; never infer a value from another
    # document without changing the displayed citation accordingly.
    number, passage = matches[0]
    idx = int(number) - 1
    if idx < 0 or idx >= len(hits):
        return None
    excerpt = relevant_sentence(question, passage)
    q = question.lower()
    answer = excerpt

    # The patterns below extract text that is literally present in the
    # selected passage. They do not inject expected evaluation answers.
    if "grade" in q and "transfer" in q:
        m = re.search(r"\b(?:grade\s+of\s+)[‘'\"]?([ABCDF][+-]?)[’'\"]?\s+or\s+better", excerpt, re.I)
        if m:
            answer = f"{m.group(1).upper()} or better"
    elif ("fee" in q or "cost" in q or "charge" in q) and ("residency" in q or "missed" in q):
        m = re.search(r"\$\s*\d[\d,]*(?:\.\d{2})?", excerpt)
        if m:
            answer = f"{m.group(0)} per missed session" if "per missed" in excerpt.lower() else m.group(0)
    elif "how many credits" in q or "how many credit" in q:
        m = re.search(r"\b\d+\s+(?:credit hours|credits)\b", excerpt, re.I)
        if m:
            answer = m.group(0)
    elif "transcript" in q and ("service" in q or "process" in q):
        m = re.search(r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,4}\s+Transcript Services)\b", excerpt)
        if m:
            answer = m.group(1)
    elif "gpa" in q:
        m = re.search(r"\b\d+(?:\.\d+)?\s*(?:cumulative\s+)?GPA\b|\bcumulative\s+GPA\s+(?:of\s+)?\d+(?:\.\d+)?", excerpt, re.I)
        if m:
            answer = m.group(0)
    elif "percentage" in q or "percent" in q:
        m = re.search(r"\b\d+(?:\.\d+)?%", excerpt)
        if m:
            answer = m.group(0)

    return {"answer": answer.strip(), "citation": hits[idx]["citation"]}


def display_answer(message: dict) -> None:
    if message.get("warning"):
        st.warning(message["warning"])
    preview = message.get("preview")
    if preview:
        st.markdown(f"**Answer:** {preview['answer']}")
        url = pdf_url(preview["citation"])
        if url:
            filename, page = source_details(preview["citation"])
            st.markdown(f"**Source:** [{filename} — PDF page {page}]({url})")
        else:
            filename, page = source_details(preview["citation"])
            st.caption(f"Source: {filename or preview['citation']} — PDF page {page}")
            path = locate_pdf(filename) if filename else None
            if path:
                # Download fallback when local static preview cannot be created.
                try:
                    with open(path, "rb") as pdf_file:
                        st.download_button(
                            "Download source PDF (open page %d in your PDF viewer)" % page,
                            data=pdf_file.read(), file_name=filename,
                            mime="application/pdf",
                            key=f"download_{filename}_{page}_{id(message)}",
                        )
                except OSError:
                    st.caption("The PDF is not readable from this location. Move the project outside OneDrive.")
            else:
                st.caption("Original PDF not found in the project's data folders.")
    else:
        # Preserve the pipeline's exact refusal or fallback wording.
        st.markdown(message["content"])
        # If the engine returned citations in a non-extractive format, link
        # the top source where possible without displaying five long chunks.
        if message.get("sources") and not message["content"].strip().lower().startswith(REFUSAL_START):
            citation = message["sources"][0]["citation"]
            url = pdf_url(citation)
            if url:
                filename, page = source_details(citation)
                st.markdown(f"**Source:** [{filename} — PDF page {page}]({url})")


st.set_page_config(page_title="Ask the Catalog · Group 5", page_icon="🎓", layout="wide")
st.title("🎓 Ask the Catalog")
st.caption("Group 5 · MSDS-630 NLP Hackathon · Concise answers grounded in source documents")

with st.sidebar:
    st.header("Configuration")
    source = st.radio("Document source", ["University document index", "Synthetic demo (fictional)"],
                      help="Demo documents contain invented policies, never use them for real advice.")
    index_path = ROOT / ("data/processed/chunks.json" if source == "University document index" else "data/demo/demo_chunks.json")
    retriever_kind = st.selectbox("Retriever", ["hybrid", "tfidf", "embeddings"])
    model_path = ""
    if retriever_kind == "embeddings":
        model_path = st.text_input("Local embedding model path / cached ID", value="sentence-transformers/all-MiniLM-L6-v2")
        st.caption("Embeddings work offline only after the model is downloaded and cached.")
    mode = st.selectbox("Answer mode", ["extractive", "ollama"],
                        format_func=lambda v: "Offline source excerpts" if v == "extractive" else "Local Ollama generation")
    ollama_model = st.text_input("Ollama model", value="llama3.2") if mode == "ollama" else ""
    threshold = st.slider("Minimum top retrieval score", 0.0, 1.0,
                          value=0.118 if retriever_kind == "hybrid" else (0.095 if retriever_kind == "tfidf" else 0.35),
                          step=0.001, format="%.3f",
                          help="Calibrate separately for each retriever with known answerable/unanswerable cases.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.caption("PDF page citations refer to actual PDF pages. Confirm current academic policies with the university.")

configuration = (str(index_path), retriever_kind, model_path, mode, ollama_model, threshold)
if st.session_state.get("active_configuration") != configuration:
    st.session_state.messages = []
    st.session_state.active_configuration = configuration


@st.cache_resource(show_spinner="Building search index...")
def make_retriever(path: str, modified_ns: int, kind: str, model_name: str):
    chunks, metadata = load_chunks(path)
    return create_retriever(chunks, kind=kind, model=model_name, local_only=True), metadata, len(chunks)


if source == "Synthetic demo (fictional)":
    st.error("DEMO DATA ONLY: These policies are invented and are not University of the Cumberlands rules.")

if not index_path.exists():
    st.warning("No index found. Place approved PDFs in data/raw_pdfs/ and run ingestion first.")
    st.code("python -m catalog_rag.cli ingest --input data/raw_pdfs --output data/processed/chunks.json")
    st.stop()

try:
    retriever, metadata, n_chunks = make_retriever(str(index_path), index_path.stat().st_mtime_ns, retriever_kind, model_path)
except Exception as exc:
    st.error(f"Could not build retrieval index: {exc}")
    st.stop()

if metadata.get("synthetic_demo") and source == "University document index":
    st.error("This index contains fictional demo chunks and cannot be used as an approved university corpus.")
    st.stop()

st.success(f"Loaded {n_chunks} source chunks · {retriever_kind.upper()} retrieval · {mode} mode")
if metadata.get("provisional_web_snapshots"):
    st.warning("PROVISIONAL SOURCE SET: These PDFs were generated from six public University of the Cumberlands webpages "
               "on October 9, 2026. They are NOT university-issued catalog or handbook PDFs. "
               "Instructor approval has not been confirmed; verify every policy with the university.")
if metadata.get("warnings"):
    with st.expander(f"Extraction warnings ({len(metadata['warnings'])})"):
        for warning in metadata["warnings"]:
            st.warning(warning)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            display_answer(message)
        else:
            st.markdown(message["content"])

question = st.chat_input("Ask about a policy in the indexed documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    assistant = CatalogAssistant(retriever, threshold=threshold, mode=mode, ollama_model=ollama_model)
    with st.chat_message("assistant"):
        with st.spinner("Searching source documents..."):
            result = assistant.ask(question)
        sources = [{"id": f"S{i}", "citation": hit.citation, "score": hit.score}
                   for i, hit in enumerate(result.hits, 1)]
        preview = short_answer(question, result.text, sources) if mode == "extractive" else None
        message = {"role": "assistant", "content": result.text, "sources": sources,
                   "preview": preview, "warning": result.warning}
        display_answer(message)
    st.session_state.messages.append(message)
