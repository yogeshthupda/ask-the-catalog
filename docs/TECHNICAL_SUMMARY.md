# Ask the Catalog — Group 5 technical summary

**MSDS 630 · October 9, 2026 · Offline-first academic-policy RAG prototype**

## Objective and implementation

Students ask natural-language academic questions; the app retrieves source PDF
passages with page citations and refuses unsupported, personal, or live-status
requests. Python 3.10+, Streamlit, PyMuPDF and scikit-learn run offline after
installation. TF-IDF baseline uses word unigrams/bigrams; the final hybrid
retriever mixes 70% word TF-IDF with 30% character n-gram TF-IDF; optional local
SentenceTransformers and local Ollama are available but **not included in these
measured results**. Extraction is verbatim with `[S1]`–`[S5]` citation IDs;
Ollama has a citation check and extractive fallback.

## Dataset and integrity

Six public official University of the Cumberlands webpages were captured on
October 9, 2026 and exported as **project-created PDF snapshots**; 39 extracted
PDF pages became 74 page-bound chunks (350 words, 50-word overlap). These are
**not university-issued catalog/handbook PDFs and are not instructor-approved**.
The 2026–27 graduate-catalog site returned HTTP 403 and the handbook required
authentication. Sources, exact URLs, and snapshot names: `data/SOURCE_PROVENANCE.md`.
25 labeled answerable questions include exact source/page, gold answer, and a
verified literal evidence anchor; five personalized, future, or live-data
questions are unanswerable. Eight separate questions set thresholds (0.095 for
TF-IDF, 0.118 for hybrid). The evaluated 30 were not a pristine held-out test:
early baseline failures informed iteration; treat final metrics as development
results, not independent generalization estimates.

## Actual metrics on the provisional snapshot corpus

| Measure | First TF-IDF baseline | Calibrated TF-IDF | Final hybrid |
|---|---:|---:|---:|
| Passage-level Recall@1, 25 answerable | 18/25 (72%) | 18/25 (72%) | 16/25 (64%) |
| Passage-level Recall@5, 25 answerable | 23/25 (92%) | 23/25 (92%) | **24/25 (96%)** |
| Correct refusals, 5 unanswerable | 3/5 | 5/5 | **5/5** |
| False refusals, 25 answerable | 2/25 | 0/25 | **0/25** |
| MRR | 0.790 | 0.790 | 0.755 |

Literal gold-evidence quote visible in final hybrid responses: **24/25**. This
objective *quote-coverage* metric does not establish semantic answer correctness
or citation faithfulness; reviewer fields remain ungraded, awaiting independent
human verification. Seventeen automated tests pass.

## Observed failure patterns and tradeoffs

1. **Q10**: master's practicum course `INTR599` in executive-resources PDF p.4
   fell outside final top-5 (the page emphasizes course codes rather than the
   exact question's wording); no gold quote appeared in the displayed answer.
2. **Q01**: nine-transfer-hour passage appeared only at rank 4; earlier
   3-passage answer display omitted it, now top-5 is shown.
3. **Q09**: seven-credit F-1 guidance was rank 4; earlier 3-passage display
   omitted it, now top-5 is shown.
4. **Q13**: Parchment was on the top retrieved page but an overly short
   sentence selector skipped its name; showing three relevant sentences plus
   neighboring context repaired quote visibility.
5. **Q15**: degree-completion instructions split across adjacent sentences;
   neighboring context now shows both deadlines.
6. Hybrid gained one Recall@5 hit but lost two first-rank hits (64% vs 72%
   Recall@1): this is a real ranking tradeoff, not a universal improvement.

## Five roles delivered / limitations

**Member 1 (Venkat, integration):** project structure, README, CLI and tests.
**Member 2 (data):** dated provenance, six PDF snapshots, 39 pages / 74 chunks.
**Member 3 (model):** baseline, hybrid, calibration and grounded/refusal logic.
**Member 4 (evaluation):** 30 gold QA labels, metrics, error and quote audit.
**Member 5 (interface):** Streamlit chatbot, source warnings, citations,
rehearsal script and ten-slide presentation. Work was AI-assisted; all team
members must still understand, independently review, and present their sections.

**Next:** get professor approval or replace snapshots with real catalog/
handbook PDFs; re-create gold references and rerun tests/evaluation. Evaluate
manual answer accuracy, accessibility and OCR/table extraction on a new blind
question set. Never use this prototype as official policy or immigration advice.
