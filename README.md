# Advanced RAG Hands-on Challenge

Three RAG tasks, each harder than the last: a baseline pipeline, then a long
structured document, then a report whose answers sit in tables, charts and
footnotes.

| Task | Folder | Document | Focus |
|---|---|---|---|
| 1 | [`task1-gpo-baseline/`](task1-gpo-baseline/) | GPO Annual Report | Ingestion, chunking, embeddings, retrieval, generation |
| 2 | [`task2-worldbank-structured/`](task2-worldbank-structured/) | World Bank Economic Report (~200 pages) | Hierarchical/section-aware chunking, metadata, context management |
| 3 | [`task3-apple-multimodal/`](task3-apple-multimodal/) | Apple iMac Environmental Report | Parsing tables, charts and footnotes so they can be retrieved |

Each task is self-contained: its own backend, dependencies, vector store and
README. Put each report PDF in that task's `data/` folder. Challenge-wide
reference material lives in [`docs/`](docs/).

## Setup

One `.env` in the repo root is shared by all tasks:

```
GOOGLE_API_KEY=your_api_key_here
```

Then follow the README inside the task folder.

## Results

| Task | Sample questions | Retrieval hit@k | Judge score |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
