# Online query path: retrieve the most relevant chunks for a question and
# answer strictly from them, citing page numbers. Stateless on purpose so the
# same question always gets the same pipeline (needed for evaluation).

from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import (
    CHAT_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    GOOGLE_API_KEY,
    NO_ANSWER_MESSAGE,
    RETRIEVAL_K,
)
from app.ingest import get_embeddings


class IndexNotBuiltError(RuntimeError):
    pass


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client, next((c for c in client.list_collections() if c.name == COLLECTION_NAME), None)


# Not cached: re-running ingest replaces the collection, and the server should
# pick that up without a restart. The completeness check stops a half-built
# index (e.g. ingest interrupted by a rate limit) from serving answers.
def _get_store() -> Chroma:
    client, collection = _get_collection()
    if collection is None:
        raise IndexNotBuiltError("Index not built yet. Run `python -m app.ingest` first.")
    total = (collection.metadata or {}).get("total_chunks")
    if total is None or collection.count() < total:
        raise IndexNotBuiltError("Index is incomplete. Re-run `python -m app.ingest` to finish it.")
    return Chroma(client=client, collection_name=COLLECTION_NAME, embedding_function=get_embeddings())


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


SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the GPO Annual Report. "
    "Answer using ONLY the numbered context excerpts below. Do not use outside knowledge "
    "and do not guess beyond what the excerpts state. Quote figures exactly as written. "
    "After each fact, cite the page it came from like (p. 12). "
    "If the excerpts do not contain the answer, respond with EXACTLY this sentence and "
    f'nothing else: "{NO_ANSWER_MESSAGE}"\n\n'
    "Context excerpts:\n{context}"
)


# Step 4: retrieve the top-k chunks with their cosine similarity.
def retrieve(question: str, k: int = RETRIEVAL_K):
    return _get_store().similarity_search_with_relevance_scores(question, k=k)


# Step 5: generate an answer grounded in the retrieved chunks.
def ask(question: str) -> dict:
    hits = retrieve(question)
    context = "\n\n---\n\n".join(
        f"[Excerpt {i} | Page {doc.metadata['page']}]\n{doc.page_content}"
        for i, (doc, _) in enumerate(hits, 1)
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
                "page": doc.metadata["page"],
                "chunk_id": doc.metadata.get("chunk_id"),
                "score": round(score, 4),
                "text": doc.page_content,
            }
            for doc, score in hits
        ],
    }
