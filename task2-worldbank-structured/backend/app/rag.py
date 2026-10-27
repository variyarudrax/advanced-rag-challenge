# Online query path: retrieve section-labelled passages for a question and
# answer strictly from them, citing section and page. Stateless on purpose so
# the same question always gets the same pipeline (needed for evaluation).

from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import CHAT_MODEL, GOOGLE_API_KEY, NO_ANSWER_MESSAGE
from app.retrieval import IndexNotBuiltError, index_status, retrieve  # noqa: F401 (re-exported)


@lru_cache(maxsize=1)
def _get_llm():
    # No temperature: gemini-3.5-flash-lite uses fixed sampling and ignores it.
    return ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GOOGLE_API_KEY)


SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the World Bank's Global Economic Prospects "
    "report (June 2026). Answer using ONLY the numbered excerpts below. Each excerpt is labelled with "
    "where it sits in the report (chapter > section > subsection) and its pages.\n"
    "- Do not use outside knowledge and do not guess beyond what the excerpts state.\n"
    "- Quote figures exactly as written, including the year they refer to.\n"
    "- When the question names a region, chapter, or box, answer from excerpts in that part of the "
    "report. Never attribute a figure from one region or chapter to another.\n"
    "- After each fact, cite its section and page like (South Asia > Outlook, p. 96).\n"
    "If the excerpts do not contain the answer, respond with EXACTLY this sentence and "
    f'nothing else: "{NO_ANSWER_MESSAGE}"\n\n'
    "Excerpts:\n{context}"
)


def _short_location(breadcrumb: str) -> str:
    # "Chapter 2: Regional Outlooks > South Asia > Outlook" -> "South Asia > Outlook";
    # chapter-level passages keep the chapter name.
    parts = breadcrumb.split(" > ")
    return " > ".join(parts[1:]) if len(parts) > 1 else parts[0]


def _pages_label(pages):
    return f"p. {pages[0]}" if len(pages) == 1 else f"pp. {pages[0]}-{pages[-1]}"


def ask(question: str) -> dict:
    where, hits, passages = retrieve(question)
    context = "\n\n---\n\n".join(
        f"[Excerpt {i} | {p['breadcrumb']} | {_pages_label(p['pages'])} | cite as: "
        f"{_short_location(p['breadcrumb'])}]\n{p['text']}"
        for i, p in enumerate(passages, 1)
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
        "route": where,
        "sources": [
            {
                "page": chunk.meta["page"],
                "pages": [int(p) for p in chunk.meta["pages"].split(",")],
                "breadcrumb": chunk.meta["breadcrumb"],
                "kind": chunk.meta["kind"],
                "chunk_id": chunk.meta["chunk_id"],
                "score": round(score, 4),
                "text": chunk.text,
            }
            for chunk, score in hits
        ],
        "context_chars": sum(len(p["text"]) for p in passages),
    }
