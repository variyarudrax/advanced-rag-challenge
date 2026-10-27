# Online query path: hybrid retrieval (dense + BM25) over text, visual, and
# footnote chunks, then an answer grounded in them that cites page and
# source type. Stateless on purpose so the same question always gets the
# same pipeline (needed for evaluation).
#
# BM25 matters more here than in Task 2: the questions ask for exact figures
# ("ENERGY STAR", "kg CO2e", "115V"), and visual chunks are short, so their
# embeddings carry little context.

import re
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from rank_bm25 import BM25Okapi

from app.config import (
    CANDIDATE_K,
    CHAT_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    GOOGLE_API_KEY,
    NO_ANSWER_MESSAGE,
    RETRIEVAL_K,
    RRF_K,
)
from app.ingest import get_embeddings


class IndexNotBuiltError(RuntimeError):
    pass


STOPWORDS = set(
    "a an and are as at be by does for from has have how in is it of on or that the this to was were what when "
    "which who why will with much many imac".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", text.lower()) if t not in STOPWORDS]


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


# All chunks in memory for BM25 (~75, tiny). Cached per index fingerprint.
@lru_cache(maxsize=1)
def _load_bm25(fingerprint: str):
    _, collection = _get_collection()
    data = collection.get(include=["documents", "metadatas"])
    chunks = list(zip(data["ids"], data["documents"], data["metadatas"]))
    return chunks, BM25Okapi([tokenize(doc) for _, doc, _ in chunks])


@lru_cache(maxsize=1)
def _get_llm():
    # No temperature: gemini-3.5-flash-lite uses fixed sampling and ignores it.
    return ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GOOGLE_API_KEY)


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


# Dense and BM25 rankings merged with reciprocal rank fusion.
# Returns [(text, metadata, fused score)] best-first.
def retrieve(question: str, k: int = RETRIEVAL_K):
    store, fingerprint = _get_store()
    chunks, bm25 = _load_bm25(fingerprint)

    dense = store.similarity_search_with_relevance_scores(question, k=CANDIDATE_K)
    dense_ids = [str(doc.metadata["chunk_id"]) for doc, _ in dense]
    scores = bm25.get_scores(tokenize(question))
    bm25_ids = [chunks[i][0] for i in sorted(range(len(chunks)), key=lambda i: -scores[i])[:CANDIDATE_K] if scores[i] > 0]

    fused: dict[str, float] = {}
    for ranking in (dense_ids, bm25_ids):
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    by_id = {cid: (doc, meta) for cid, doc, meta in chunks}
    best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    return [(*by_id[cid], score) for cid, score in best]


SYSTEM_PROMPT = (
    "You are an assistant that answers questions about Apple's iMac Product Environmental Report "
    "(October 2024). Answer using ONLY the numbered excerpts below. Excerpts come from body text, "
    "charts, tables, callouts, logos, and endnotes; each is labelled with its page and type.\n"
    "- Do not use outside knowledge and do not guess beyond what the excerpts state.\n"
    "- Quote figures exactly as written, with their units and the product configuration they apply to "
    "(e.g. iMac (two ports) 256GB). If the report gives figures for more than one configuration, give each.\n"
    "- If a footnote qualifies a figure (how it was measured, what it excludes), mention it briefly.\n"
    "- After each fact, cite its page and source type like (p. 10, table) or (p. 12, endnote 18).\n"
    "If the excerpts do not contain the answer, respond with EXACTLY this sentence and "
    f'nothing else: "{NO_ANSWER_MESSAGE}"\n\n'
    "Excerpts:\n{context}"
)


def ask(question: str) -> dict:
    hits = retrieve(question)
    context = "\n\n---\n\n".join(
        f"[Excerpt {i} | p. {meta['page']} | {meta['kind']}]\n{doc}" for i, (doc, meta, _) in enumerate(hits, 1)
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.replace("{context}", context)),
        HumanMessage(content=question),
    ]
    content = _get_llm().invoke(messages).content
    # Gemini may return a list of content parts instead of a plain string.
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    answer = content.strip()

    return {
        "answer": answer,
        "found": answer != NO_ANSWER_MESSAGE,
        "sources": [
            {
                "page": meta["page"],
                "kind": meta["kind"],
                "title": meta["title"],
                "footnotes": [int(n) for n in meta["footnotes"].split(",") if n],
                "chunk_id": meta["chunk_id"],
                "score": round(score, 4),
                "text": doc,
            }
            for doc, meta, score in hits
        ],
    }
