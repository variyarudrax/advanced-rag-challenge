# Offline ingestion: build chunks from the cached extraction (see
# app.extract), embed them, and persist the vector store to chroma_db/.
#
#     python -m app.extract   # once: text layer, endnotes, vision
#     python -m app.ingest    # chunk + embed
#
# Three kinds of chunk:
#   text      body paragraphs from the text layer, split per page
#   visual    one per chart / table / callout / logo / diagram from vision
#   footnote  one per endnote, including any table printed inside it
#
# Footnote linking: a chunk that cites "[^18]" gets the text of endnote 18
# appended. A claim and its qualifier ("58% less energy than ENERGY STAR"
# and the power table that backs it) are then retrieved together, even
# though they sit ten pages apart.

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
    DOC_TITLE,
    EMBED_BATCH_SIZE,
    EMBED_REQUESTS_PER_MINUTE,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    find_pdf,
)
from app.extract import load_extracted

# Bump when the chunk format changes, so old indexes are rebuilt.
CHUNK_VERSION = "1"

MARKER_RE = re.compile(r"\[\^(\d{1,2})\]")


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)


def _footnote_text(number: int, footnotes: dict, note_tables: dict) -> str:
    note = footnotes.get(number)
    if not note:
        return ""
    tables = "".join(f"\n{t}" for t in note_tables.get(number, []))
    return f"[^{number}] {note['text']}{tables}"


def _with_footnotes(body: str, refs: list[int], footnotes: dict, note_tables: dict) -> str:
    notes = [_footnote_text(n, footnotes, note_tables) for n in sorted(set(refs))]
    notes = [n for n in notes if n]
    return body + ("\n\nFootnotes:\n" + "\n".join(notes) if notes else "")


def build_chunks(pages, footnotes, source):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    # Tables printed inside an endnote (the p.12 power table in note 18).
    note_tables: dict[int, list[str]] = {}
    for p in pages:
        for e in p["vision"]:
            if e.get("in_footnote"):
                note_tables.setdefault(e["in_footnote"], []).append(f"{e['title']}\n{e['content']}")

    chunks = []

    def add(kind, page, title, body, refs):
        header = f"[{DOC_TITLE} > p. {page} {title}]"
        chunks.append(
            Document(
                page_content=f"{header}\n{_with_footnotes(body, refs, footnotes, note_tables)}",
                metadata={
                    "source": source,
                    "chunk_id": len(chunks),
                    "page": page,
                    "kind": kind,
                    "title": title,
                    "footnotes": ",".join(map(str, sorted(set(refs)))),
                },
            )
        )

    for p in pages:
        page, title = p["page"], p["title"]
        if not p["endnotes_page"]:
            # Chart labels repeat ("Materials and process emissions" x3);
            # the vision chunk carries the chart, so drop repeats here.
            seen, blocks = set(), []
            for t in p["text"]:
                if t not in seen and t != title:
                    seen.add(t)
                    blocks.append(t)
            for part in splitter.split_text("\n\n".join(blocks)):
                add("text", page, title, part, [int(n) for n in MARKER_RE.findall(part)])
        for e in p["vision"]:
            label = f"{title} > {e['kind'].upper()}: {e['title']}"
            add(e["kind"], page, label, e["content"], e["footnotes"])

    for number, note in sorted(footnotes.items()):
        # The note itself (with any table inside it) is the body; notes it
        # cites in turn are appended like any other reference.
        body = _footnote_text(number, footnotes, note_tables)
        refs = [int(n) for n in MARKER_RE.findall(note["text"]) if int(n) != number]
        add("footnote", note["page"], f"Endnotes > Endnote {number}", body, refs)
    return chunks


def _fingerprint(pdf_path, chunks) -> str:
    # Extraction output can change without the PDF changing (vision re-run),
    # so the chunk count and total size are part of the fingerprint.
    size = sum(len(c.page_content) for c in chunks)
    return f"{pdf_path.name}:{CHUNK_SIZE}:{CHUNK_OVERLAP}:{EMBEDDING_MODEL}:c{CHUNK_VERSION}:{len(chunks)}:{size}"


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


def build_index(chunks, fingerprint, fresh=False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = next((c for c in client.list_collections() if c.name == COLLECTION_NAME), None)
    if existing is not None and (fresh or (existing.metadata or {}).get("fingerprint") != fingerprint):
        print("  discarding previous index (settings or extraction changed)")
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
    parser.add_argument("--dry-run", action="store_true", help="build chunks only; no embedding calls")
    args = parser.parse_args()

    pdf_path = find_pdf()
    pages, footnotes = load_extracted()
    chunks = build_chunks(pages, footnotes, pdf_path.name)
    kinds = {}
    for c in chunks:
        kinds[c.metadata["kind"]] = kinds.get(c.metadata["kind"], 0) + 1
    lengths = [len(c.page_content) for c in chunks]
    print(f"{len(chunks)} chunks (avg {sum(lengths) // len(lengths)} chars, max {max(lengths)}) {kinds}")
    if args.dry_run:
        return

    print("Embedding and indexing ...")
    build_index(chunks, _fingerprint(pdf_path, chunks), fresh=args.fresh)
    print(f"Done. Index saved to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
