# Offline ingestion: load the GPO Annual Report PDF, split it into chunks,
# embed them, and persist the vector store to chroma_db/. Run once (or again
# after changing chunking settings):
#
#     python -m app.ingest

import argparse
import re
import sys
import time

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBED_REQUESTS_PER_MINUTE,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    find_pdf,
)


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)


def _clean(text: str) -> str:
    # Rejoin words hyphenated across line breaks and collapse runs of blank
    # lines/spaces left over from PDF layout.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Step 1: load the PDF one Document per page. PyMuPDF handles multi-column
# layouts and tables better than pypdf. Page numbers are stored 1-indexed so
# citations match what a reader sees in a PDF viewer.
def load_pages(pdf_path):
    pages = PyMuPDFLoader(str(pdf_path)).load()
    docs = []
    for page in pages:
        page.page_content = _clean(page.page_content)
        if not page.page_content:
            continue
        page.metadata = {
            "source": pdf_path.name,
            "page": page.metadata.get("page", 0) + 1,
        }
        docs.append(page)
    return docs


# Step 2: split pages into overlapping chunks. Chunks never cross a page
# boundary, so every chunk has exactly one page to cite.
def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


# Identifies what the index was built from. If any of these change, the
# stored chunks are stale and the index is rebuilt from scratch.
def _fingerprint(pdf_path) -> str:
    return f"{pdf_path.name}:{pdf_path.stat().st_size}:{CHUNK_SIZE}:{CHUNK_OVERLAP}:{EMBEDDING_MODEL}"


# Embeds one batch, waiting out per-minute 429s. A per-day quota can't be
# waited out, so stop and let the next run resume from where this one ended.
def _add_with_retry(store, batch, max_retries=8):
    for attempt in range(max_retries):
        try:
            store.add_documents(batch, ids=[str(c.metadata["chunk_id"]) for c in batch])
            return
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" not in msg and "429" not in msg:
                raise
            if "PerDay" in msg:
                sys.exit("\nDaily embedding quota exhausted. Progress is saved; re-run `python -m app.ingest` later to resume.")
            match = re.search(r"retry in ([\d.]+)s", msg)
            wait = float(match.group(1)) + 2 if match else 30 * (attempt + 1)
            print(f"  rate limited, waiting {wait:.0f}s ...")
            time.sleep(wait)
    sys.exit("\nStill rate limited after retries. Progress is saved; re-run to resume.")


# Step 3: embed the chunks and persist them. Resumes a previous partial run
# when its fingerprint matches; otherwise (or with --fresh) starts over.
def build_index(chunks, fingerprint, fresh=False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = next((c for c in client.list_collections() if c.name == COLLECTION_NAME), None)
    if existing is not None and (fresh or (existing.metadata or {}).get("fingerprint") != fingerprint):
        print("  discarding previous index (settings or PDF changed)")
        client.delete_collection(COLLECTION_NAME)

    store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        collection_metadata={
            "hnsw:space": "cosine",
            "fingerprint": fingerprint,
            "total_chunks": len(chunks),
        },
    )
    done = set(store.get(include=[])["ids"])
    pending = [c for c in chunks if str(c.metadata["chunk_id"]) not in done]
    if done:
        print(f"  resuming: {len(done)} chunks already embedded, {len(pending)} to go")

    # Free tier counts each chunk as one request (limit 100/min), so pace
    # batches to stay under EMBED_REQUESTS_PER_MINUTE.
    for start in range(0, len(pending), EMBED_BATCH_SIZE):
        batch = pending[start : start + EMBED_BATCH_SIZE]
        began = time.monotonic()
        _add_with_retry(store, batch)
        print(f"  embedded {len(done) + start + len(batch)}/{len(chunks)} chunks")
        if start + len(batch) < len(pending):
            min_gap = 60 * len(batch) / EMBED_REQUESTS_PER_MINUTE
            time.sleep(max(0, min_gap - (time.monotonic() - began)))
    return store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="discard any existing index and rebuild")
    args = parser.parse_args()

    pdf_path = find_pdf()
    print(f"Loading {pdf_path.name} ...")
    pages = load_pages(pdf_path)
    print(f"  {len(pages)} pages with text")

    chunks = chunk_pages(pages)
    lengths = [len(c.page_content) for c in chunks]
    print(f"  {len(chunks)} chunks (avg {sum(lengths) // len(lengths)} chars, max {max(lengths)})")

    print("Embedding and indexing ...")
    build_index(chunks, _fingerprint(pdf_path), fresh=args.fresh)
    print(f"Done. Index saved to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
