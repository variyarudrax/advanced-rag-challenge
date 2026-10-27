# GPO Annual Report – Baseline RAG

Ask questions about the GPO Annual Report and get answers grounded only in the
report, with page citations. Built on the AskGPT stack.

- **Backend:** Python, FastAPI, LangChain, Gemini (`gemini-3.5-flash-lite` for chat, `gemini-embedding-001` for embeddings), Chroma (persisted to `chroma_db/`)
- **Frontend:** Vite + React

## Pipeline

| Step | Where | Baseline choice |
|---|---|---|
| 1. Ingestion | `backend/app/ingest.py` | PyMuPDF, one document per page, light text cleanup, 1-indexed page metadata |
| 2. Chunking | `backend/app/ingest.py` | Recursive splitter, 1000 chars / 150 overlap, chunks never cross pages |
| 3. Embeddings | `backend/app/ingest.py` | `gemini-embedding-001`, cosine space, persisted Chroma index |
| 4. Retrieval | `backend/app/rag.py` | Top-k similarity search, k = 6 |
| 5. Generation | `backend/app/rag.py` | Grounded prompt, cites `(p. N)`, fixed "not found" reply |
| Validation | `backend/app/evaluate.py` | Runs sample questions → CSV/JSON in `results/`; page hit@k + optional LLM judge |

All tuning knobs are in `backend/app/config.py`.

## Setup

1. Put the report PDF in `task1-gpo-baseline/data/` (any filename ending in `.pdf`).
2. Put the sample questions in `data/sample_questions.json`, using the format in `data/sample_questions.example.json`. Only `question` is required. `expected_answer` enables `--judge`, and `expected_pages` enables the retrieval hit rate.
3. `.env` in the repo root (shared by all tasks):
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Run

```bash
cd task1-gpo-baseline/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m app.ingest                      # build the index (once)
python -m app.evaluate --judge            # validate on sample questions
uvicorn app.main:app --reload --port 8000 # API for the UI
```

```bash
cd task1-gpo-baseline/frontend
npm install
npm run dev        # http://localhost:5173
```

## API

- `GET /status`: whether the index is built, document name, chunk count
- `GET /sample-questions`: questions from `data/sample_questions.json`
- `POST /ask` `{"question": "..."}` → `{answer, found, sources: [{page, chunk_id, score, text}]}`

## Notes

- Re-run `python -m app.ingest` after changing chunk settings or the PDF.
- Free-tier Gemini quotas are small. `evaluate.py --delay` spaces out requests.
