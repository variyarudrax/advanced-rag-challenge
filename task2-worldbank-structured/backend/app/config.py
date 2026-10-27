# Central config: API key, model names, paths, and chunking/retrieval tuning.
# Everything an experiment might change lives here.
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
# One .env at the repo root is shared by all tasks.
load_dotenv(ROOT_DIR.parent / ".env")

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# Same models as Task 1 so results are comparable.
CHAT_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Paths. The report PDF is read from data/; override with WB_PDF_PATH.
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
RESULTS_DIR = ROOT_DIR / "results"
SAMPLE_QUESTIONS_PATH = DATA_DIR / "sample_questions.json"
COLLECTION_NAME = "world_bank_gep"


def find_pdf() -> Path:
    override = os.environ.get("WB_PDF_PATH")
    if override:
        return Path(override)
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {DATA_DIR}. Put the World Bank report there.")
    return pdfs[0]


# Chunking. Chunks stay inside one section but may span pages, so they can be
# a little larger than Task 1's page-bound chunks.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Retrieval. Dense and BM25 each return CANDIDATE_K hits, fused with
# reciprocal rank fusion; the top RETRIEVAL_K seed the context.
CANDIDATE_K = 20
RETRIEVAL_K = 6
RRF_K = 60
# Context management: each seed chunk pulls in this many neighbours on each
# side (same section only), and the whole context is capped at this size.
NEIGHBOR_WINDOW = 1
CONTEXT_CHAR_BUDGET = 14000
# If a metadata filter leaves fewer chunks than this, fall back to the full
# index rather than answer from a tiny slice.
MIN_FILTERED_CHUNKS = 3

# Embedding requests are sent in batches and paced: the free tier counts each
# chunk as one request and allows 100/min, so stay a little under that.
EMBED_BATCH_SIZE = 50
EMBED_REQUESTS_PER_MINUTE = 90

# Exact string the LLM must return when the report doesn't answer the question.
NO_ANSWER_MESSAGE = "I couldn't find that information in the Global Economic Prospects report."
