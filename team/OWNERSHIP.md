# Group 5 — Team charter and integration handoffs

**Completion update, October 9, 2026:** The AI-assisted project package now
contains completed artifacts for all five role workstreams: integration, six
provisional university-webpage PDF snapshots / 74 chunks, offline retrieval and
calibration, a 30-question source-labeled evaluation, a Streamlit demonstration,
a ten-slide deck and a one-page technical report. Specific deliverables are
documented in the updated `README.md`. Role names 2–5 are deliberately blank
because their identities were never supplied. **Artifact creation by the
assistant is not proof that the human members completed/reviewed the work.**
The team must each check the code, labels, sources, and assigned slides. The
professor must approve these substitute PDFs or provide the actual catalog/
handbook PDFs for a complete re-evaluation. The evaluation's semantic
correctness and citation faithfulness have not received independent student
judgments; the 24/25 automated metric is literal quote visibility only.

Fill names and phone/communication channels at kickoff. All 5 speak on Sunday.

| Person | Name | Main artifact | Integration handoff | Talk time |
|---|---|---|---|---|
| 1 — Lead | Venkat | Repo, integration, README | Confirm CLI/app ready and data is the approved set | 0–1 min, 1–3 min shared |
| 2 — Data | __________ | `ingest.py` and `chunks.json` | JSON `{chunk_id,document,page,text}`; review table extraction | 1–3 min shared |
| 3 — Modeling | __________ | `retrieval.py`, `pipeline.py` | `search(question,k)` returns hits; `ask(question)` returns response | 3–6 min shared |
| 4 — Evaluation | __________ | `questions.csv`, results, error analysis | Frozen 30 QA labels and actual metrics; manual audit | 6–8 min, 8–10 min shared |
| 5 — Product/demo | __________ | `app.py`, slide coordination | UI uses `CatalogAssistant`; scripted live demo and fallback | 3–6 min shared |

## Fixed team rules

1. Public, instructor-approved university PDFs only; label document edition and PDF page numbers.
2. Ask 30 questions (25 answerable, 5 unanswerable) **before model tuning**.
3. No model predictions or metrics may be fabricated, extrapolated or sourced from fictional demo results.
4. Do not give legal or immigration/personalized approval advice or claim the prototype is official.
5. Every teammate can explain every submitted component and records how AI assistance was used.
6. Five-minute stand-up at the start of each session; push/backup at least every two hours.

## Handoffs — minimize integration conflicts

- Member 2 exports `data/processed/chunks.json` in the schema produced by `save_chunks` and notes any image-only pages.
- Member 3 does not change the `Chunk`, `Hit`, or `Answer` public fields without coordinating with Members 4 and 5.
- Member 4 verifies gold answers directly against PDFs and shares a locked evaluation CSV; never reverse-engineers labels from model output.
- Member 5 tests long questions, empty questions, missing index, Ollama offline, and refusal display.
- Venkat runs tests, both CLI commands, and a complete dry-run on the **presenting laptop** before code freeze.

## Weekend milestones

- **Friday Oct 9:** 30-minute charter, confirm PDFs, 30 gold QAs, extracted chunks, TF-IDF baseline, first evaluation number.
- **Saturday Oct 10 (morning/midday):** add chat UI + optional LLM; instructor checkpoint demonstration.
- **Saturday Oct 10 (afternoon/evening):** benchmark, manual error analysis, one-page technical summary, ≤10 slides, freeze/submission at instructor's announced time.
- **Sunday Oct 11:** rehearse 10-minute slot, 5-minute Q&A; all five speak.

## Demo questions to prepare (replace after real corpus review)

1. One simple academic policy question with verified document/page.
2. A paraphrased version that challenges word overlap (and may favor embeddings).
3. A truly unanswerable question, demonstrating refusal.

Always have terminal CLI and pre-recorded screenshots of *actual verified output* available as a fallback if the UI or local LLM fails. Do not pass off a screenshot as a live result.

