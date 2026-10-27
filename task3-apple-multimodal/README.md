# Apple iMac Environmental Report – Multimodal Document RAG

Ask questions about Apple's *iMac Product Environmental Report* (October
2024, 12 pages) and get answers that draw on its charts, tables, callouts,
logos, and footnotes, not only its body text. Same stack as Tasks 1–2.

- **Backend:** Python, FastAPI, LangChain, Gemini (`gemini-3.5-flash-lite` for chat **and vision**, `gemini-embedding-001` for embeddings), Chroma, BM25 (`rank_bm25`)
- **Frontend:** Vite + React (shows the evidence type of every source)

## The problem

The report is short, but its key facts are not in paragraphs:

| Fact | Where it lives | Why plain text extraction fails |
|---|---|---|
| 445 kg baseline → −99 kg → 346 kg | Waterfall chart (p. 2) | Text layer is a jumble of repeated bar labels; rule-based table detection reads the chart as a table |
| Power draw by mode and voltage | Table **inside endnote 18** (p. 12) | Striped rows split it into fragments; it is ten pages from the "58% less energy" claim it backs |
| 346 kg CO2e: 50% production, 45% use | Two-column stat table (p. 10) | Detected as three unrelated one-row tables |
| EPEAT Gold | Logo (p. 1) + endnote 6 | The logo has no text at all |
| 100% recycled aluminum stand, 76% recycled packaging | Callouts and icon captions (pp. 1, 6) | Scattered short blocks with no reading order |

## Pipeline

| Step | Approach | Where |
|---|---|---|
| 1. Text layer | PyMuPDF blocks; superscript footnote markers rewritten as `[^18]`; page titles from the largest font | `app/extract.py` |
| 2. Endnotes | Parsed into `{number: {page, text}}`; number-heavy blocks (table cells) excluded | `app/extract.py` |
| 3. Vision | Each page rendered at 150 dpi and sent to Gemini with its text layer. Structured output: one element per chart / table / callout / logo / diagram, with exact printed values, footnote refs, and whether it sits inside an endnote. Prompt forbids estimating unlabeled values | `app/extract.py` |
| Cache | Per-page JSON + `document.md` in `extracted/` (committed), so vision runs once and the extraction can be reviewed | `extracted/` |
| 4. Chunks | `text` (body, split per page), one chunk per visual element, one per endnote (75 total) | `app/ingest.py` |
| 5. Footnote linking | A chunk citing `[^N]` gets endnote N appended, including any table printed inside it | `app/ingest.py` |
| 6. Retrieval | Dense + BM25, reciprocal rank fusion, top 6 | `app/rag.py` |
| 7. Generation | Cites page and evidence type, e.g. `(p. 10, table)`, `(p. 12, endnote 18)`; gives figures per product configuration | `app/rag.py` |
| Validation | Page hit, **evidence hit** (a chunk of the expected kind from the expected page was retrieved), LLM judge | `app/evaluate.py` |

All tuning knobs are in `backend/app/config.py`.

## Results

`data/sample_questions.json` has the three questions from the use-case brief
plus four added ones (`"added": true`) whose answers exist only in a chart, a
table inside a footnote, an endnote, or a callout.

| Metric | Result |
|---|---|
| Page hit@6 | 7/7 |
| Evidence hit@6 | 7/7 |
| LLM judge | 7/7 CORRECT |

Known issues: citations sometimes name the wrong evidence type or page for a
correct fact (a table figure cited as "text"; an endnote cited with the page
of the claim rather than p. 11).

## Setup

1. The report PDF goes in `data/` (any filename ending in `.pdf`).
2. `.env` in the repo root (shared by all tasks):
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Run

```bash
cd task3-apple-multimodal/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m app.extract                     # only needed if extracted/ is missing or the PDF changed (12 vision calls)
python -m app.ingest                      # build the index (75 chunks)
python -m app.evaluate --judge            # validate on sample questions
uvicorn app.main:app --reload --port 8000 # API for the UI
```

```bash
cd task3-apple-multimodal/frontend
npm install
npm run dev        # http://localhost:5173
```

## API

- `GET /status`: whether the index is built, document name, chunk count
- `GET /sample-questions`: questions from `data/sample_questions.json`
- `POST /ask` `{"question": "..."}` → `{answer, found, sources: [{page, kind, title, footnotes, chunk_id, score, text}]}`

`kind` is one of `text`, `chart`, `table`, `callout`, `logo`, `diagram`, `footnote`.

## Notes

- `python -m app.extract --refresh` re-runs vision on every page; review
  `extracted/document.md` afterwards. Vision output is not deterministic.
- The index fingerprint includes the chunk count and size, so re-running
  extract and then ingest rebuilds the index when the extraction changed.
- Scores in `sources` are reciprocal-rank-fusion scores, not cosine similarity.
