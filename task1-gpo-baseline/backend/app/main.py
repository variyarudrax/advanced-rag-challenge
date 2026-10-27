# FastAPI app exposing the GPO Annual Report RAG to the React frontend.
# The index is built offline by app.ingest; this server only answers questions.

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.config import SAMPLE_QUESTIONS_PATH
from app.rag import IndexNotBuiltError, ask, index_status

app = FastAPI(title="GPO Annual Report RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.get("/status")
async def status():
    return await run_in_threadpool(index_status)


# Returns the sample questions (if provided) so the UI can offer them as
# one-click prompts.
@app.get("/sample-questions")
async def sample_questions():
    if not SAMPLE_QUESTIONS_PATH.exists():
        return {"questions": []}
    items = json.loads(SAMPLE_QUESTIONS_PATH.read_text())
    return {"questions": [item["question"] for item in items]}


@app.post("/ask")
async def ask_question(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    try:
        # Blocking embedding + LLM calls; keep them off the event loop.
        return await run_in_threadpool(ask, req.question.strip())
    except IndexNotBuiltError as e:
        raise HTTPException(503, str(e))
