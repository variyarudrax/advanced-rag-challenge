# Validation: run every sample question through the pipeline and write a
# results table to results/. Run after ingest:
#
#     python -m app.evaluate            # answers + retrieval page/section hits
#     python -m app.evaluate --judge    # also grade answers with an LLM judge
#
# data/sample_questions.json format (only "question" is required):
# [
#   {"question": "...", "expected_answer": "...", "expected_pages": [12, 13],
#    "expected_section": "South Asia"}
# ]
#
# expected_section is matched against each retrieved chunk's breadcrumb
# ("Chapter 2: Regional Outlooks > South Asia > Outlook"), so it can name a
# chapter, section, subsection, or box.

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
        retrieved_pages = sorted({p for s in result["sources"] for p in s["pages"]})
        expected_pages = item.get("expected_pages") or []
        expected = item.get("expected_answer", "")
        expected_section = item.get("expected_section")
        # Share of retrieved chunks that come from the expected section: low
        # values mean the right answer is being diluted by other regions or
        # chapters even when it is retrieved.
        in_section = [expected_section in s["breadcrumb"] for s in result["sources"]] if expected_section else []

        row = {
            "id": i,
            "question": question,
            "expected_answer": expected,
            "answer": result["answer"],
            "found": result["found"],
            "route": json.dumps(result["route"]) if result["route"] else "",
            "retrieved_pages": retrieved_pages,
            "retrieved_sections": [s["breadcrumb"] for s in result["sources"]],
            "top_score": result["sources"][0]["score"] if result["sources"] else None,
            "context_chars": result["context_chars"],
            "expected_pages": expected_pages,
            "page_hit": bool(set(expected_pages) & set(retrieved_pages)) if expected_pages else None,
            "expected_section": expected_section,
            "section_hit": any(in_section) if expected_section else None,
            "section_precision": round(sum(in_section) / len(in_section), 2) if in_section else None,
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
    section_hits = [r["section_hit"] for r in rows if r["section_hit"] is not None]
    precisions = [r["section_precision"] for r in rows if r["section_precision"] is not None]
    verdicts = [r["verdict"] for r in rows if r["verdict"]]

    print("\n===== Summary =====")
    print(f"Questions:          {n}")
    print(f"Answered (not 'not found'): {answered}/{n}")
    if hits:
        print(f"Retrieval page hit@k: {sum(hits)}/{len(hits)} ({sum(hits) / len(hits):.0%})")
    if section_hits:
        print(f"Section hit@k:        {sum(section_hits)}/{len(section_hits)} ({sum(section_hits) / len(section_hits):.0%})")
        print(f"Section precision:    {sum(precisions) / len(precisions):.0%} of retrieved chunks from the expected section")
    if verdicts:
        for label in ("CORRECT", "PARTIAL", "INCORRECT", "UNKNOWN"):
            print(f"{label:<10} {verdicts.count(label)}")
    print(f"\nResults written to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
