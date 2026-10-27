# World Bank Global Economic Prospects – Structured Document RAG

Ask questions about the World Bank's *Global Economic Prospects* (June 2026,
200 pages) and get answers grounded in the right chapter and section, cited
by section and page. Same stack as Task 1 so the two are directly comparable.

- **Backend:** Python, FastAPI, LangChain, Gemini (`gemini-3.5-flash-lite` for chat, `gemini-embedding-001` for embeddings), Chroma, BM25 (`rank_bm25`)
- **Frontend:** Vite + React (Task 1's UI, showing section paths and the search route)

## What changes from Task 1

Task 1 treats the report as a flat list of pages. That breaks down on a long
report where every region has its own "Recent developments / Outlook /
Risks" section saying similar things: "What is the growth outlook for South
Asia?" matches East Asia's or Sub-Saharan Africa's outlook almost as well.

| Step | Task 1 (baseline) | Task 2 (this) | Where |
|---|---|---|---|
| Parsing | Page text | Document hierarchy recovered from typography: chapter › box › section › subsection, plus body / figure / table. Drops running headers, contents pages, reference lists. Repairs the PDF's broken "Th" and "fi/fl" ligatures | `app/parse.py` |
| Chunking | 1000 chars, never cross a page | 1200 chars, never cross a **section** (may cross pages; all pages kept). Tables are chunked separately and repeat their column-header row | `app/ingest.py` |
| Chunk text | Raw text | Prefixed with its section path, e.g. `[Chapter 2: Regional Outlooks > South Asia > Outlook]` | `app/ingest.py` |
| Metadata | page | chapter, section, subsection, box, kind, label, page range, segment position | `app/ingest.py` |
| Retrieval | Dense top-6 | **Routing**: a question naming a region, chapter, box, or the executive summary is searched only within that part. **Hybrid**: dense + BM25, merged with reciprocal rank fusion | `app/retrieval.py` |
| Context | 6 chunks as-is | Each hit pulls in its neighbouring chunks from the same section; runs are merged into section-labelled passages within a 14k-character budget | `app/retrieval.py` |
| Generation | Cites `(p. N)` | Cites `(South Asia > Outlook, p. 96)`; told never to carry a figure across regions or chapters | `app/rag.py` |
| Validation | Page hit, LLM judge | Adds **section hit** and **section precision** (share of retrieved chunks from the expected section) | `app/evaluate.py` |

All tuning knobs are in `backend/app/config.py`.

## How the structure is recovered

The PDF has bookmarks only for chapters, boxes, and regions, so the finer
levels come from fonts (measured with PyMuPDF):

| Element | Typography |
|---|---|
| Section | Helvetica Bold 16pt ("Summary", "South Asia") |
| Subsection | Helvetica Bold 14pt, or Helvetica Roman 11pt in color ("Recent developments", "Outlook") |
| Box / Figure / Table | Helvetica Bold 10–11pt in color ("BOX 1.1", "FIGURE 2.7", "TABLE 2.9") |
| Body text | Garamond 11pt in the main text, 10.6pt inside boxes |
| Footnotes | Garamond under 9.5pt |

Boxes carry a "BOX x.y (continued)" label on every page, so box text is kept
apart from the chapter text running beside it. To see the recovered outline
(no API calls):

```bash
python -m app.parse
```

## Setup

1. The report PDF goes in `data/` (any filename ending in `.pdf`).
2. Sample questions are in `data/sample_questions.json`. Besides the Task 1
   fields, `expected_section` names the chapter/section/box the answer should
   come from.
3. `.env` in the repo root (shared by all tasks):
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Run

```bash
cd task2-worldbank-structured/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m app.ingest --dry-run            # parse + chunk only, no API calls
python -m app.ingest                      # build the index (~620 chunks, ~7 min on the free tier; resumes)
python -m app.evaluate --judge            # validate on sample questions
uvicorn app.main:app --reload --port 8000 # API for the UI
```

```bash
cd task2-worldbank-structured/frontend
npm install
npm run dev        # http://localhost:5173
```

## API

- `GET /status`: whether the index is built, document name, chunk count
- `GET /sample-questions`: questions from `data/sample_questions.json`
- `POST /ask` `{"question": "..."}` → `{answer, found, route, context_chars, sources: [{page, pages, breadcrumb, kind, chunk_id, score, text}]}`

`route` is the metadata filter the question was routed to (e.g.
`{"section": "South Asia"}`), or `null` when the whole report was searched.

## Notes

- Re-run `python -m app.ingest` after changing parsing or chunk settings; the
  index fingerprint includes `PARSER_VERSION` in `ingest.py`.
- Routing is rule-based (aliases in `retrieval.py`), so it is deterministic and
  free, but only recognizes the names it lists. If a filter matches fewer than
  `MIN_FILTERED_CHUNKS` chunks, the whole report is searched instead.
- Scores in `sources` are reciprocal-rank-fusion scores, not cosine similarity.
