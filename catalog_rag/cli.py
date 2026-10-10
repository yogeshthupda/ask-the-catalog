"""Run: python -m catalog_rag.cli --help"""
from __future__ import annotations

import argparse
import json

from .evaluation import read_questions, run_evaluation, write_results
from .ingest import chunk_pages, read_pdfs
from .pipeline import CatalogAssistant
from .retrieval import create_retriever
from .storage import load_chunks, save_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Group 5 Ask the Catalog tools")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Extract and chunk locally stored PDFs")
    ingest.add_argument("--input", default="data/raw_pdfs")
    ingest.add_argument("--output", default="data/processed/chunks.json")
    ingest.add_argument("--chunk-words", type=int, default=350)
    ingest.add_argument("--overlap-words", type=int, default=50)

    validate = sub.add_parser("validate-eval", help="Validate the 30-question evaluation CSV")
    validate.add_argument("--questions", default="evaluation/questions_template.csv")

    for name in ("ask", "evaluate"):
        p = sub.add_parser(name, help="Ask one question" if name == "ask" else "Run evaluation")
        p.add_argument("--index", default="data/processed/chunks.json")
        p.add_argument("--retriever", choices=["tfidf", "hybrid", "embeddings"], default="tfidf")
        p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
        p.add_argument("--allow-download", action="store_true", help="Allow fetching optional embedding model")
        p.add_argument("--threshold", type=float, default=0.12)
        p.add_argument("--mode", choices=["extractive", "ollama"], default="extractive")
        p.add_argument("--ollama-model", default="llama3.2")
        if name == "ask":
            p.add_argument("--question", required=True)
        else:
            p.add_argument("--questions", default="evaluation/questions_template.csv")
            p.add_argument("--output-prefix", default="evaluation/results/tfidf")

    args = parser.parse_args()
    if args.command == "ingest":
        pages, warnings = read_pdfs(args.input)
        chunks = chunk_pages(pages, args.chunk_words, args.overlap_words)
        save_chunks(args.output, chunks, metadata={
            "source_directory": args.input,
            "source_documents": sorted({p.document for p in pages}),
            "provisional_web_snapshots": bool(pages) and all(
                p.document.startswith("UC_Web_Snapshot_") for p in pages
            ),
            "chunk_words": args.chunk_words, "overlap_words": args.overlap_words,
            "warnings": warnings,
        })
        print(f"Wrote {len(chunks)} chunks from {len(pages)} PDF pages to {args.output}")
        for warning in warnings:
            print("WARNING:", warning)
        return
    if args.command == "validate-eval":
        questions = read_questions(args.questions)
        print(f"Validated {len(questions)} questions (25 answerable, 5 unanswerable)")
        return

    chunks, metadata = load_chunks(args.index)
    if metadata.get("synthetic_demo"):
        print("WARNING: This corpus is fictional DEMO DATA, not university policy.")
    if metadata.get("provisional_web_snapshots"):
        print("WARNING: This corpus uses project-created PDFs of public university webpages;"
              " it is NOT a professor-approved catalog/handbook collection.")
    retriever = create_retriever(chunks, args.retriever, model=args.model, local_only=not args.allow_download)
    assistant = CatalogAssistant(retriever, threshold=args.threshold, mode=args.mode,
                                 ollama_model=args.ollama_model)
    if args.command == "ask":
        answer = assistant.ask(args.question)
        print(answer.text)
        if answer.warning:
            print("WARNING:", answer.warning)
    elif args.command == "evaluate":
        questions = read_questions(args.questions)
        summary, rows = run_evaluation(assistant, questions)
        write_results(args.output_prefix, summary, rows)
        print(json.dumps(summary, indent=2))
        print(f"Wrote evaluation files with prefix: {args.output_prefix}")


if __name__ == "__main__":
    main()

