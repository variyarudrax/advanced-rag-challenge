# Validation: run every sample question through the pipeline and write a
# results table to results/. Run after ingest:
#
#     python -m app.evaluate            # answers + retrieval page/evidence hits
#     python -m app.evaluate --judge    # also grade answers with an LLM judge
#
# data/sample_questions.json format (only "question" is required):
# [
#   {"question": "...", "expected_answer": "...", "expected_pages": [10],
#    "expected_kind": "table"}
# ]
#
# expected_kind is where the answer actually lives: text, chart, table,
# callout, logo, diagram, or footnote. evidence_hit is true when a retrieved
# chunk of that kind comes from one of the expected pages: the check that a
# text-only pipeline would fail.

import argparse
import csv
import json
import time
from datetime import datetime

from langchain_core.messages import HumanMessage

from app.config import RESULTS_DIR, SAMPLE_QUESTIONS_PATH
from app.rag import _get_llm, ask

JUDGE_PROMPT = """You are grading a question-answering system.
Question: {question}
Expected answer: {expected}
System answer: {answer}

Is the system answer factually consistent with the expected answer and does it
contain its key facts? Reply with exactly one word: CORRECT, PARTIAL, or INCORRECT."""


def judge(question, expected, answer) -> str:
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, answer=answer)
    content = _get_llm().invoke([HumanMessage(content=prompt)]).content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    verdict = content.strip().upper()
    for label in ("INCORRECT", "PARTIAL", "CORRECT"):
        if label in verdict:
            return label
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="grade answers with an LLM judge")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between questions (rate limits)")
    args = parser.parse_args()

    if not SAMPLE_QUESTIONS_PATH.exists():
        raise SystemExit(
            f"{SAMPLE_QUESTIONS_PATH} not found. Create it in the format of sample_questions.example.json."
        )
    items = json.loads(SAMPLE_QUESTIONS_PATH.read_text())
    rows = []
    for i, item in enumerate(items, 1):
        question = item["question"]
        print(f"[{i}/{len(items)}] {question}")
        result = ask(question)
        sources = result["sources"]
        retrieved_pages = sorted({s["page"] for s in sources})
        expected_pages = item.get("expected_pages") or []
        expected = item.get("expected_answer", "")
        expected_kind = item.get("expected_kind")

        row = {
            "id": i,
            "question": question,
            "expected_answer": expected,
            "answer": result["answer"],
            "found": result["found"],
            "retrieved": [f"p{s['page']}:{s['kind']}" for s in sources],
            "top_score": sources[0]["score"] if sources else None,
            "expected_pages": expected_pages,
            "page_hit": bool(set(expected_pages) & set(retrieved_pages)) if expected_pages else None,
            "expected_kind": expected_kind,
            "evidence_hit": (
                any(s["kind"] == expected_kind and (not expected_pages or s["page"] in expected_pages) for s in sources)
                if expected_kind
                else None
            ),
            "verdict": judge(question, expected, result["answer"]) if args.judge and expected else None,
        }
        rows.append(row)
        time.sleep(args.delay)

    write_results(rows)


def write_results(rows):
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = RESULTS_DIR / f"eval_{stamp}.json"
    json_path.write_text(json.dumps(rows, indent=2))

    csv_path = RESULTS_DIR / f"eval_{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Summary metrics.
    n = len(rows)
    answered = sum(r["found"] for r in rows)
    hits = [r["page_hit"] for r in rows if r["page_hit"] is not None]
    evidence = [r["evidence_hit"] for r in rows if r["evidence_hit"] is not None]
    verdicts = [r["verdict"] for r in rows if r["verdict"]]

    print("\n===== Summary =====")
    print(f"Questions:          {n}")
    print(f"Answered (not 'not found'): {answered}/{n}")
    if hits:
        print(f"Retrieval page hit@k: {sum(hits)}/{len(hits)} ({sum(hits) / len(hits):.0%})")
    if evidence:
        print(f"Evidence hit@k:       {sum(evidence)}/{len(evidence)} ({sum(evidence) / len(evidence):.0%}) right chart/table/footnote retrieved")
    if verdicts:
        for label in ("CORRECT", "PARTIAL", "INCORRECT", "UNKNOWN"):
            print(f"{label:<10} {verdicts.count(label)}")
    print(f"\nResults written to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
