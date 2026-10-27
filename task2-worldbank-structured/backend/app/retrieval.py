# Retrieval for a long, structured report, in three steps:
#
# 1. Route: if the question names a part of the report ("South Asia",
#    "Chapter 3", "Box 1.1", "executive summary"), restrict the search to
#    chunks from that part, using the metadata written by ingest.
# 2. Search: dense (embedding) and BM25 (keyword) search run side by side and
#    are merged with reciprocal rank fusion. BM25 catches exact terms the
#    embedding blurs: country names, table numbers, "SAR".
# 3. Expand: each hit pulls in its neighbouring chunks from the same section,
#    and the result is grouped into section-labelled passages that fit a
#    fixed character budget.

import re
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from app.config import (
    CANDIDATE_K,
    CHROMA_DIR,
    COLLECTION_NAME,
    CONTEXT_CHAR_BUDGET,
    MIN_FILTERED_CHUNKS,
    NEIGHBOR_WINDOW,
    RETRIEVAL_K,
    RRF_K,
)
from app.ingest import get_embeddings


class IndexNotBuiltError(RuntimeError):
    pass


# Region names and the abbreviations the report itself uses. Bare "Middle
# East" is left out on purpose: the conflict there is mentioned everywhere.
REGION_ALIASES = {
    "East Asia and Pacific": ["east asia and pacific", "east asia", "eap"],
    "Europe and Central Asia": ["europe and central asia", "central asia", "eca"],
    "Latin America and the Caribbean": ["latin america", "caribbean", "lac"],
    "Middle East, North Africa, Afghanistan, and Pakistan": ["north africa", "mena", "mna"],
    "South Asia": ["south asia", "sar"],
    "Sub-Saharan Africa": ["sub-saharan africa", "sub saharan africa", "ssa"],
}

STOPWORDS = set(
    "a an and are as at be by does for from has have how in is it of on or that the this to was were what when "
    "which who why will with according say says report".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    id: str
    text: str
    meta: dict


@dataclass
class Index:
    chunks: list[Chunk]
    by_id: dict[str, Chunk]
    by_seg: dict[tuple[int, int], Chunk]
    bm25: BM25Okapi


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client, next((c for c in client.list_collections() if c.name == COLLECTION_NAME), None)


# Not cached: re-running ingest replaces the collection, and the server should
# pick that up without a restart. The completeness check stops a half-built
# index (e.g. ingest interrupted by a rate limit) from serving answers.
def _get_store() -> tuple[Chroma, str]:
    client, collection = _get_collection()
    if collection is None:
        raise IndexNotBuiltError("Index not built yet. Run `python -m app.ingest` first.")
    total = (collection.metadata or {}).get("total_chunks")
    if total is None or collection.count() < total:
        raise IndexNotBuiltError("Index is incomplete. Re-run `python -m app.ingest` to finish it.")
    store = Chroma(client=client, collection_name=COLLECTION_NAME, embedding_function=get_embeddings())
    return store, collection.metadata["fingerprint"]


# BM25 and neighbour lookup need every chunk in memory (~600, a few MB).
# Cached per index fingerprint, so a rebuilt index is reloaded.
@lru_cache(maxsize=1)
def _load_index(fingerprint: str) -> Index:
    _, collection = _get_collection()
    data = collection.get(include=["documents", "metadatas"])
    chunks = [Chunk(i, d, m) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])]
    chunks.sort(key=lambda c: c.meta["chunk_id"])
    return Index(
        chunks=chunks,
        by_id={c.id: c for c in chunks},
        by_seg={(c.meta["seg_id"], c.meta["seg_pos"]): c for c in chunks},
        bm25=BM25Okapi([tokenize(c.text) for c in chunks]),
    )


# Step 1: map a question to a metadata filter, or None for the whole report.
def route(question: str, index: Index) -> dict | None:
    q = question.lower()
    values = lambda field: sorted({c.meta[field] for c in index.chunks if c.meta[field]})

    if m := re.search(r"\bbox\s+(\d+\.\d+)", q):
        box = next((b for b in values("box") if b.startswith(f"Box {m.group(1)}:")), None)
        if box:
            return {"box": box}
    if m := re.search(r"\bchapter\s+(\d)\b", q):
        chapter = next((c for c in values("chapter") if c.startswith(f"Chapter {m.group(1)}:")), None)
        if chapter:
            return {"chapter": chapter}
    if "executive summary" in q:
        return {"section": "Executive Summary"}
    if "foreword" in q:
        return {"section": "Foreword"}
    if "statistical appendix" in q:
        return {"chapter": "Statistical Appendix"}
    regions = [
        region
        for region, aliases in REGION_ALIASES.items()
        if any(re.search(rf"\b{re.escape(a)}\b", q) for a in aliases)
    ]
    if len(regions) == 1:
        return {"section": regions[0]}
    if len(regions) > 1:
        return {"$or": [{"section": r} for r in regions]}
    return None


def _matches(meta: dict, where: dict | None) -> bool:
    if where is None:
        return True
    if "$or" in where:
        return any(_matches(meta, w) for w in where["$or"])
    return all(meta.get(k) == v for k, v in where.items())


# Step 2: hybrid search. Returns (chunk, fused score) best-first.
def search(question: str, index: Index, store: Chroma, where: dict | None, k: int):
    dense = store.similarity_search_with_relevance_scores(question, k=CANDIDATE_K, filter=where)
    dense_ids = [str(doc.metadata["chunk_id"]) for doc, _ in dense]

    scores = index.bm25.get_scores(tokenize(question))
    allowed = [i for i, c in enumerate(index.chunks) if _matches(c.meta, where)]
    bm25_ids = [index.chunks[i].id for i in sorted(allowed, key=lambda i: -scores[i])[:CANDIDATE_K] if scores[i] > 0]

    fused: dict[str, float] = {}
    for ranking in (dense_ids, bm25_ids):
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    return [(index.by_id[cid], score) for cid, score in best]


def _merge_overlap(a: str, b: str) -> str:
    # Neighbouring chunks overlap by up to CHUNK_OVERLAP characters.
    for size in range(min(len(a), len(b), 400), 20, -1):
        if a.endswith(b[:size]):
            return a + b[size:]
    return a + " " + b


def _body(chunk: Chunk) -> str:
    # Drop the "[breadcrumb]" line; passages carry their own label.
    return chunk.text.split("\n", 1)[1] if chunk.text.startswith("[") else chunk.text


# Step 3: expand each hit with its neighbours, merge runs of consecutive
# chunks from one segment into a passage, and keep passages (best hit first)
# until the character budget is spent. Returned in document order.
def build_context(hits, index: Index) -> list[dict]:
    passages: dict[int, set[int]] = {}
    rank_of: dict[int, int] = {}
    for rank, (chunk, _) in enumerate(hits):
        seg, pos = chunk.meta["seg_id"], chunk.meta["seg_pos"]
        positions = passages.setdefault(seg, set())
        for p in range(pos - NEIGHBOR_WINDOW, pos + NEIGHBOR_WINDOW + 1):
            if (seg, p) in index.by_seg:
                positions.add(p)
        rank_of.setdefault(seg, rank)

    selected, used = [], 0
    for seg in sorted(passages, key=lambda s: rank_of[s]):
        run = [index.by_seg[(seg, p)] for p in sorted(passages[seg])]
        text = _body(run[0])
        for prev, nxt in zip(run, run[1:]):
            gap = nxt.meta["seg_pos"] - prev.meta["seg_pos"] > 1
            text = text + "\n[...]\n" + _body(nxt) if gap else _merge_overlap(text, _body(nxt))
        if used + len(text) > CONTEXT_CHAR_BUDGET and selected:
            continue
        used += len(text)
        pages = sorted({int(p) for c in run for p in c.meta["pages"].split(",")})
        selected.append(
            {
                "breadcrumb": run[0].meta["breadcrumb"],
                "kind": run[0].meta["kind"],
                "pages": pages,
                "text": text,
                "order": run[0].meta["chunk_id"],
            }
        )
    return sorted(selected, key=lambda p: p["order"])


def retrieve(question: str, k: int = RETRIEVAL_K):
    store, fingerprint = _get_store()
    index = _load_index(fingerprint)
    where = route(question, index)
    if where is not None and sum(_matches(c.meta, where) for c in index.chunks) < MIN_FILTERED_CHUNKS:
        where = None
    hits = search(question, index, store, where, k)
    return where, hits, build_context(hits, index)


def index_status() -> dict:
    _, collection = _get_collection()
    if collection is None:
        return {"ready": False, "document": None, "chunks": 0, "total_chunks": None}
    count = collection.count()
    total = (collection.metadata or {}).get("total_chunks")
    sample = collection.peek(1)["metadatas"]
    return {
        "ready": total is not None and count >= total,
        "document": sample[0]["source"] if sample else None,
        "chunks": count,
        "total_chunks": total,
    }
