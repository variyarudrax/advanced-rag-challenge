# Offline ingestion: parse the report into structured segments, chunk each
# segment, embed the chunks, and persist the vector store to chroma_db/.
# Run once (or again after changing parsing/chunking settings):
#
#     python -m app.ingest

import argparse
import re
import sys
import time

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
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
from app.parse import parse_pdf

# Bump when parse.py or the chunk format changes, so old indexes are rebuilt.
PARSER_VERSION = "1"

YEARS_RE = re.compile(r"(20\d\d\D+){3,}")


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)


# Body paragraphs broken across columns or pages are rejoined with a space;
# finished paragraphs get a blank line between them.
def _join_text(pieces):
    text, starts = "", []
    for piece in pieces:
        if text:
            text += "\n\n" if re.search(r"[.!?:”\")]$", text) else " "
        starts.append((len(text), piece.page))
        text += piece.text
    return text, starts


def _join_graphic(pieces):
    text, starts = "", []
    for piece in pieces:
        if text:
            text += "\n\n"
        starts.append((len(text), piece.page))
        text += piece.text
    return text, starts


def _pages_for(starts, begin, end):
    pages = [page for offset, page in starts if offset < end]
    first = next((page for offset, page in reversed(starts) if offset <= begin), pages[0])
    return sorted({p for p in pages if p >= first})


def _table_header(text):
    # Lines up to and including the column-year row ("2023 2024 2025e ...").
    lines = text.split("\n")
    for i, line in enumerate(lines[:12]):
        if YEARS_RE.search(line + " "):
            return "\n".join(lines[: i + 1])
    return lines[0]


# Step 2: chunk inside each segment, so a chunk never spans two sections.
# Every chunk starts with its breadcrumb ("Chapter 2: Regional Outlooks >
# South Asia > Outlook"), so both the embedding and BM25 see where it sits.
def chunk_segments(segments, source):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = []
    for seg_id, seg in enumerate(segments):
        join = _join_text if seg.kind == "text" else _join_graphic
        text, starts = join(seg.pieces)
        parts = splitter.create_documents([text])
        header = _table_header(text) if seg.kind == "table" else None
        for pos, part in enumerate(parts):
            begin = part.metadata["start_index"]
            body = part.page_content
            # Continued table chunks repeat the column header so numbers keep
            # their meaning.
            if header and pos > 0 and not body.startswith(header):
                body = f"{header}\n...\n{body}"
            pages = _pages_for(starts, begin, begin + len(part.page_content))
            chunks.append(
                Document(
                    page_content=f"[{seg.breadcrumb}]\n{body}",
                    metadata={
                        "source": source,
                        "chunk_id": len(chunks),
                        "seg_id": seg_id,
                        "seg_pos": pos,
                        "chapter": seg.chapter,
                        "section": seg.section or "",
                        "subsection": seg.subsection or "",
                        "box": seg.box or "",
                        "kind": seg.kind,
                        "label": seg.label or "",
                        "breadcrumb": seg.breadcrumb,
                        "page": pages[0],
                        "page_end": pages[-1],
                        "pages": ",".join(map(str, pages)),
                    },
                )
            )
    return chunks


# Identifies what the index was built from. If any of these change, the
# stored chunks are stale and the index is rebuilt from scratch.
def _fingerprint(pdf_path) -> str:
    return f"{pdf_path.name}:{pdf_path.stat().st_size}:{CHUNK_SIZE}:{CHUNK_OVERLAP}:{EMBEDDING_MODEL}:p{PARSER_VERSION}"


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
    parser.add_argument("--dry-run", action="store_true", help="parse and chunk only; no embedding calls")
    args = parser.parse_args()

    pdf_path = find_pdf()
    print(f"Parsing {pdf_path.name} ...")
    segments = parse_pdf(pdf_path)
    print(f"  {len(segments)} segments")

    chunks = chunk_segments(segments, pdf_path.name)
    lengths = [len(c.page_content) for c in chunks]
    kinds = {}
    for c in chunks:
        kinds[c.metadata["kind"]] = kinds.get(c.metadata["kind"], 0) + 1
    print(f"  {len(chunks)} chunks (avg {sum(lengths) // len(lengths)} chars, max {max(lengths)}) {kinds}")
    if args.dry_run:
        return

    print("Embedding and indexing ...")
    build_index(chunks, _fingerprint(pdf_path), fresh=args.fresh)
    print(f"Done. Index saved to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
