# Central config: API key, model names, paths, and chunking/retrieval tuning.
# Everything an experiment might change lives here.
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
# One .env at the repo root is shared by all tasks.
load_dotenv(ROOT_DIR.parent / ".env")

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# Same models as Tasks 1-2 so results are comparable. The chat model is
# multimodal, so it also reads the page images during extraction.
CHAT_MODEL = "gemini-3.5-flash-lite"
VISION_MODEL = CHAT_MODEL
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Paths. The report PDF is read from data/; override with IMAC_PDF_PATH.
# extracted/ holds the per-page extraction (text, footnotes, vision output).
# It is committed so the vision step does not have to be re-run to review or
# rebuild the index.
DATA_DIR = ROOT_DIR / "data"
EXTRACTED_DIR = ROOT_DIR / "extracted"
CHROMA_DIR = ROOT_DIR / "chroma_db"
RESULTS_DIR = ROOT_DIR / "results"
SAMPLE_QUESTIONS_PATH = DATA_DIR / "sample_questions.json"
COLLECTION_NAME = "imac_environmental_report"
DOC_TITLE = "iMac Product Environmental Report"


def find_pdf() -> Path:
    override = os.environ.get("IMAC_PDF_PATH")
    if override:
        return Path(override)
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {DATA_DIR}. Put the iMac Environmental Report there.")
    return pdfs[0]


# Page images sent to the vision model. 150 dpi keeps 5pt footnote markers
# and table digits legible.
RENDER_DPI = 150

# Chunking. The report is 12 short pages; body text is split per page.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

# Retrieval: dense and BM25 each return CANDIDATE_K hits, fused with
# reciprocal rank fusion; the top RETRIEVAL_K go to the LLM.
CANDIDATE_K = 15
RETRIEVAL_K = 6
RRF_K = 60

EMBED_BATCH_SIZE = 50
EMBED_REQUESTS_PER_MINUTE = 90

# Exact string the LLM must return when the report doesn't answer the question.
NO_ANSWER_MESSAGE = "I couldn't find that information in the iMac Product Environmental Report."
