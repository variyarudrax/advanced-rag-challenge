# Offline extraction: turn each page of the iMac report into searchable
# pieces, from three sources:
#
# 1. Text layer (PyMuPDF): body paragraphs, with superscript footnote markers
#    rewritten as "[^18]" so they survive chunking.
# 2. Endnotes (pages titled "Endnotes"): parsed into {number: text}.
# 3. Vision (Gemini): each page is rendered to an image and the model
#    transcribes every chart, table, callout, logo, and diagram into text.
#    Rule-based table detection fails on this report: it reads the p.2
#    waterfall chart as a table and splits the striped p.12 power table into
#    fragments, and logos (EPEAT Gold) have no text at all.
#
# Results are cached in extracted/ (one JSON per page, footnotes.json, and a
# readable document.md), so the vision calls run once:
#
#     python -m app.extract            # extract pages not yet cached
#     python -m app.extract --refresh  # redo every page

import argparse
import base64
import json
import re
import sys
import time
from typing import Literal

import pymupdf
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.config import DOC_TITLE, EXTRACTED_DIR, GOOGLE_API_KEY, RENDER_DPI, VISION_MODEL, find_pdf

HEADER_RE = re.compile(r"^iMac\s*\|\s*Product Environmental Report")
# "1 Product recycled ..." or "[^3]Breakdown ..." (some markers are superscript).
ENDNOTE_RE = re.compile(r"^\s*(?:\[\^(\d{1,2})\]|(\d{1,2})\s)\s*(.+)", re.S)


# --- text layer -------------------------------------------------------------


def _block_text(block) -> str:
    lines = []
    for line in block.get("lines", []):
        parts = []
        for s in line["spans"]:
            t = s["text"]
            # Superscript digits are footnote markers ("ENERGY STAR.18").
            if s["flags"] & 1 and t.strip().isdigit():
                parts.append(f"[^{t.strip()}]")
            else:
                parts.append(t)
        lines.append("".join(parts).strip())
    text = ""
    for line in lines:
        if not line:
            continue
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        else:
            text = f"{text} {line}" if text else line
    return re.sub(r"\s+", " ", text).strip()


def _page_title(page) -> str:
    # The page title is the first block set in the page's largest font
    # (e.g. "Use", "Progress toward carbon neutral"); it may wrap.
    spans = [
        (bi, s)
        for bi, b in enumerate(page.get_text("dict")["blocks"])
        for line in b.get("lines", [])
        for s in line["spans"]
        if s["text"].strip()
    ]
    if not spans:
        return ""
    top = max(s["size"] for _, s in spans)
    first_block = next(bi for bi, s in spans if s["size"] >= top - 0.5)
    return " ".join(s["text"].strip() for bi, s in spans if bi == first_block and s["size"] >= top - 0.5)


def text_blocks(page) -> list[str]:
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        text = _block_text(b)
        if not text or HEADER_RE.match(text) or text.isdigit():
            continue
        blocks.append(text)
    return blocks


def parse_endnotes(blocks: list[str]) -> dict[int, str]:
    notes, current = {}, None
    for text in blocks:
        if text == "Endnotes" or text.startswith("©"):
            continue
        m = ENDNOTE_RE.match(text)
        number = int(m.group(1) or m.group(2)) if m else None
        # Numbers must increase, so "1) No use of ..." inside note 7 is not note 1.
        if number is not None and number not in notes and (current is None or number > current):
            current = number
            notes[current] = m.group(3).strip()
        elif current is not None and len(text) >= 40 and sum(c.isdigit() for c in text) / len(text) < 0.15:
            # Continuation paragraphs and bullets. Short or number-heavy
            # blocks are cells of a table printed inside a note (the p.12
            # power table); vision transcribes those whole.
            notes[current] += "\n" + text
    return notes


# --- vision -----------------------------------------------------------------


class VisualElement(BaseModel):
    kind: Literal["chart", "table", "callout", "logo", "diagram"]
    title: str = Field(description="Short title: the element's printed heading, or a descriptive one.")
    content: str = Field(
        description="Everything the element says, as text. Tables as full markdown tables. "
        "Charts: every printed value with its label, then a plain description of what the chart shows."
    )
    footnotes: list[int] = Field(default_factory=list, description="Footnote numbers referenced in the element.")
    in_footnote: int | None = Field(
        default=None, description="If the element is printed inside an endnote, that endnote's number."
    )


class PageVisuals(BaseModel):
    elements: list[VisualElement]


VISION_PROMPT = """You are extracting information from page {page} of Apple's "{title}" so it can be searched by a question-answering system.

The page's text layer is below. Plain body paragraphs are already captured from it, so do NOT repeat them.
Extract every element whose meaning depends on layout or graphics:
- chart: title, every value printed on it with its label and unit, then one or two sentences on what it shows
  (what the bars/segments are and how they relate). NEVER estimate a value that is not printed; say "not labeled".
- table: the complete table as markdown, every row and column, values and units exactly as printed.
- callout: short facts set apart from body text (big numbers, highlighted claims, icon captions, side panels).
- logo: certification marks or badges, with what they certify (e.g. an EPEAT Gold registration mark).
- diagram: process or life-cycle diagrams, listing their stages.
Footnote references are small superscript numbers; list them in `footnotes`.
If an element is printed inside an endnote (endnote pages), set `in_footnote` to that endnote's number.
Return an empty list if the page has no such elements.

Text layer of the page:
---
{text}
---"""


def _vision_llm():
    return ChatGoogleGenerativeAI(model=VISION_MODEL, google_api_key=GOOGLE_API_KEY).with_structured_output(
        PageVisuals
    )


def describe_page(llm, page, page_no: int, text: str, max_retries=6) -> list[dict]:
    png = page.get_pixmap(dpi=RENDER_DPI).tobytes("png")
    message = HumanMessage(
        content=[
            {"type": "text", "text": VISION_PROMPT.format(page=page_no, title=DOC_TITLE, text=text)},
            {"type": "image_url", "image_url": f"data:image/png;base64,{base64.b64encode(png).decode()}"},
        ]
    )
    for attempt in range(max_retries):
        try:
            result = llm.invoke([message])
            return [e.model_dump() for e in result.elements] if result else []
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" not in msg and "429" not in msg:
                raise
            if "PerDay" in msg:
                sys.exit("\nDaily quota exhausted. Finished pages are cached; re-run later to resume.")
            match = re.search(r"retry in ([\d.]+)s", msg)
            wait = float(match.group(1)) + 2 if match else 20 * (attempt + 1)
            print(f"  rate limited, waiting {wait:.0f}s ...")
            time.sleep(wait)
    sys.exit("\nStill rate limited after retries. Finished pages are cached; re-run to resume.")


# --- output -----------------------------------------------------------------


def _page_path(page_no: int):
    return EXTRACTED_DIR / f"page_{page_no:02d}.json"


def load_extracted() -> tuple[list[dict], dict[int, dict]]:
    pages = [json.loads(p.read_text()) for p in sorted(EXTRACTED_DIR.glob("page_*.json"))]
    if not pages:
        raise FileNotFoundError(f"Nothing in {EXTRACTED_DIR}. Run `python -m app.extract` first.")
    footnotes = {int(k): v for k, v in json.loads((EXTRACTED_DIR / "footnotes.json").read_text()).items()}
    return pages, footnotes


def write_markdown(pages, footnotes):
    out = [f"# {DOC_TITLE}: extracted content\n"]
    for p in pages:
        out.append(f"\n## Page {p['page']}: {p['title']}\n")
        if not p["endnotes_page"]:
            out.extend(f"{t}\n" for t in p["text"])
        for e in p["vision"]:
            refs = f" (footnotes {', '.join(map(str, e['footnotes']))})" if e["footnotes"] else ""
            where = f" [inside endnote {e['in_footnote']}]" if e.get("in_footnote") else ""
            out.append(f"\n**[{e['kind']}] {e['title']}**{refs}{where}\n\n{e['content']}\n")
    out.append("\n## Endnotes\n")
    out.extend(f"\n[^{n}]: (p. {f['page']}) {f['text']}\n" for n, f in sorted(footnotes.items()))
    (EXTRACTED_DIR / "document.md").write_text("".join(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-run vision on every page")
    args = parser.parse_args()

    pdf_path = find_pdf()
    doc = pymupdf.open(str(pdf_path))
    EXTRACTED_DIR.mkdir(exist_ok=True)
    llm = _vision_llm()

    pages, footnotes = [], {}
    for index, page in enumerate(doc):
        page_no = index + 1
        title = _page_title(page)
        blocks = text_blocks(page)
        endnotes_page = title == "Endnotes"
        if endnotes_page:
            for number, text in parse_endnotes(blocks).items():
                footnotes[number] = {"page": page_no, "text": text}

        path = _page_path(page_no)
        cached = json.loads(path.read_text()) if path.exists() else None
        if cached and not args.refresh:
            vision = cached["vision"]
            print(f"p.{page_no:<2} {title[:40]:<40} cached ({len(vision)} visual elements)")
        else:
            vision = describe_page(llm, page, page_no, "\n".join(blocks))
            print(f"p.{page_no:<2} {title[:40]:<40} {len(vision)} visual elements")

        record = {
            "page": page_no,
            "title": title,
            "endnotes_page": endnotes_page,
            "text": blocks,
            "vision": vision,
        }
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        pages.append(record)

    (EXTRACTED_DIR / "footnotes.json").write_text(json.dumps(footnotes, indent=2, ensure_ascii=False))
    write_markdown(pages, footnotes)
    print(f"\n{len(footnotes)} endnotes. Wrote {EXTRACTED_DIR}/ (see document.md)")


if __name__ == "__main__":
    main()
