# Central config: API key, model names, paths, and chunking/retrieval tuning.
# Everything a baseline experiment might change lives here.
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
# One .env at the repo root is shared by all tasks.
load_dotenv(ROOT_DIR.parent / ".env")

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# Same models as AskGPT (pinned after a live smoke test there).
CHAT_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Paths. The report PDF is read from data/; override with GPO_PDF_PATH.
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
RESULTS_DIR = ROOT_DIR / "results"
SAMPLE_QUESTIONS_PATH = DATA_DIR / "sample_questions.json"
COLLECTION_NAME = "gpo_annual_report"


def find_pdf() -> Path:
    override = os.environ.get("GPO_PDF_PATH")
    if override:
        return Path(override)
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {DATA_DIR}. Put the GPO Annual Report there.")
    return pdfs[0]


# Chunking + retrieval tuning. Annual reports are dense, so retrieve a few
# more chunks than AskGPT's k=4.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 6

# Embedding requests are sent in batches and paced: the free tier counts each
# chunk as one request and allows 100/min, so stay a little under that.
EMBED_BATCH_SIZE = 50
EMBED_REQUESTS_PER_MINUTE = 90

# Exact string the LLM must return when the report doesn't answer the question.
NO_ANSWER_MESSAGE = "I couldn't find that information in the GPO Annual Report."
