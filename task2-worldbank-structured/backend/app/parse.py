# Structure-aware PDF parsing for the World Bank Global Economic Prospects
# report. Turns the PDF into "segments": runs of text that share one place in
# the document hierarchy (chapter > box > section > subsection) and one kind
# (body text, figure, or table). Chunking then happens inside segments, so a
# chunk never mixes, say, South Asia with Sub-Saharan Africa.
#
# The report has no usable heading tags, so the hierarchy is recovered from
# typography (font family, size, color), measured once with PyMuPDF:
#
#   running header   HelveticaLTStd-Light 7pt at the top ("CHAPTER 2 ...")
#   chapter opener   Garamond 26-30pt ("CHAPTER 1" / "GLOBAL OUTLOOK")
#   front matter     HelveticaLTStd-Cond 26pt ("Foreword", "Executive Summary")
#   section          HelveticaLTStd-Bold 16pt ("Summary", "South Asia")
#   subsection       HelveticaLTStd-Bold 14pt, or HelveticaLTStd-Roman 11pt in
#                    color ("Recent developments", "Outlook")
#   box/figure/table HelveticaLTStd-Bold 10-11pt in color ("BOX 1.1 ...",
#                    "FIGURE 2.7 ...", "TABLE 2.9 ...")
#   body text        Garamond 11pt in the main flow, 10.6pt inside boxes
#   footnotes        Garamond under 9.5pt
#   everything else  chart labels, axis ticks, table cells (small Helvetica)
#
# Run on its own to inspect the recovered outline without any API calls:
#
#     python -m app.parse

import re
from dataclasses import dataclass, field

import pymupdf

from app.config import find_pdf

# Front-matter pages that only add noise to retrieval (dot leaders, acronyms).
SKIP_FRONT_SECTIONS = {"Contents", "Summary of Contents", "Abbreviations", "Acknowledgments"}
# Bibliographies and publication lists match many queries on title words but
# never answer them.
SKIP_SECTIONS_RE = re.compile(r"^(References|Global Economic Prospects: Selected Topics|Prospects Group: Selected)")
WHITE = 16777215

HEADER_RE = re.compile(r"^(CHAPTER\s+(\d)|STATISTICAL APPENDIX)")
LABEL_RE = re.compile(r"^(BOX|FIGURE|TABLE)\s+([A-Z]?\d[\d.]*)\s*(.*)$")
NUMERIC_RE = re.compile(r"^[\d.,\-–—%()\s]+$")

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
# The embedded font drops the "Th" ligature, so "The" is extracted as "Te".
# Only words that are not English on their own are repaired ("Ten" is left).
TH_RE = re.compile(r"\bT(e|is|ese|ey|at|eir|ere|ose|us|ough|roughout|ird|ree|em)\b")


def normalize(text: str) -> str:
    for lig, rep in LIGATURES.items():
        text = text.replace(lig, rep)
    text = TH_RE.sub(lambda m: "Th" + m.group(1), text)
    text = text.replace("­", "")
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Piece:
    page: int
    text: str


@dataclass
class Segment:
    chapter: str
    section: str | None
    subsection: str | None
    box: str | None
    kind: str  # "text" | "figure" | "table"
    label: str | None  # e.g. "TABLE 2.9 South Asia country forecasts"
    pieces: list[Piece] = field(default_factory=list)

    @property
    def key(self):
        return (self.chapter, self.box, self.section, self.subsection, self.kind, self.label)

    @property
    def breadcrumb(self) -> str:
        parts = [self.chapter, self.box, self.section, self.subsection, self.label]
        return " > ".join(p for p in parts if p)

    @property
    def pages(self) -> list[int]:
        return sorted({p.page for p in self.pieces})


# --- block classification -------------------------------------------------


@dataclass
class Block:
    x0: float
    y0: float
    font: str
    size: float
    color: int
    text: str
    spans: list  # (x0, y0, text) for graphic text, kept for row grouping


def _join_lines(lines: list[str]) -> str:
    # Rejoin words hyphenated at line ends ("com-" + "modity").
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}" if out else line
    return out


def _blocks(page) -> list[Block]:
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        lines, spans_xy, first = [], [], None
        for line in b.get("lines", []):
            # Drop superscript footnote markers (tiny spans inside body lines).
            spans = [s for s in line["spans"] if s["text"].strip() and s["size"] >= 6]
            if not spans:
                continue
            first = first or spans[0]
            lines.append("".join(s["text"] for s in spans))
            spans_xy.extend((s["bbox"][0], s["bbox"][1], s["text"]) for s in spans)
        if first is None:
            # Footnotes are set in 4.8-9pt; keep their text even if every span is small.
            small = [s for line in b.get("lines", []) for s in line["spans"] if s["text"].strip()]
            if not small:
                continue
            first = small[0]
            lines = ["".join(s["text"] for s in line["spans"]) for line in b["lines"]]
        x0, y0 = b["bbox"][:2]
        blocks.append(
            Block(x0, y0, first["font"], round(first["size"], 1), first["color"], _join_lines(lines), spans_xy)
        )
    return blocks


def _classify(b: Block, page_height: float) -> str:
    f, s = b.font, b.size
    if f.startswith("HelveticaLTStd-Light") and b.y0 < 60:
        return "header"
    if f.startswith("HelveticaLTStd-Bold") and s <= 8.5 and b.y0 > page_height - 70:
        return "footer"
    if f.startswith("HelveticaLTStd-Cond") and s >= 20:
        return "front_heading"
    if s >= 20:
        return "chapter_title"
    if b.color == WHITE and s >= 11.5:
        return "skip"  # cover and back-cover blurbs (table header bars are smaller)
    if f.startswith("HelveticaLTStd-Bold") and s >= 15:
        return "skip" if "continued" in b.text.lower() else "h2"
    if f.startswith("HelveticaLTStd-Bold") and 13 <= s < 15:
        return "h3"
    if f.startswith("HelveticaLTStd-Roman") and 10.5 <= s < 12 and b.color != 0:
        return "h3"
    if f.startswith("HelveticaLTStd-Bold") and 9.5 <= s <= 11.5 and b.color != 0 and LABEL_RE.match(b.text):
        return "label"
    if f.startswith(("AGaramondPro", "MinionPro")):
        return "body" if s >= 9.5 else "footnote"
    return "graphic"


# --- graphics (figures and tables) ----------------------------------------


def _rows(spans) -> list[str]:
    # Group spans into visual rows (same baseline within 2.5pt), left to right.
    rows, current, last_y = [], [], None
    for x, y, t in sorted(spans, key=lambda s: (round(s[1]), s[0])):
        if last_y is not None and abs(y - last_y) > 2.5:
            rows.append(current)
            current = []
        current.append((x, t))
        last_y = y
    if current:
        rows.append(current)
    return [" ".join(t.strip() for _, t in sorted(r)) for r in rows]


def _graphic_text(label_text: str, blocks: list[Block], is_table: bool) -> str:
    rows = _rows([s for b in blocks for s in b.spans])
    if not is_table:
        # Axis ticks and bare data labels carry no meaning without the plot.
        rows = [r for r in rows if not NUMERIC_RE.match(r) and len(r) > 3]
    return normalize(label_text) + "\n" + "\n".join(normalize(r) for r in rows if r.strip())


def _assign_graphics(labels: list[Block], graphics: list[Block], mid: float):
    # Each graphic block belongs to the nearest label above it in the same
    # column (labels that start in the left column of a full-width figure
    # count for both). Blocks above every label go to the first label.
    groups = {id(lb): [] for lb in labels}
    for g in graphics:
        same_col = [lb for lb in labels if lb.y0 <= g.y0 + 2 and (lb.x0 < mid - 20 or g.x0 >= mid - 20)]
        target = max(same_col, key=lambda lb: lb.y0) if same_col else min(labels, key=lambda lb: lb.y0)
        groups[id(target)].append(g)
    return groups


# --- document walk ----------------------------------------------------------


def _chapter_titles(doc) -> dict[str, str]:
    titles = {}
    for _, title, _ in doc.get_toc():
        m = re.match(r"Ch\.\s*(\d)\s+(.*)", title, re.S)
        if m:
            titles[m.group(1)] = " ".join(m.group(2).split())
    return titles


def parse_pdf(pdf_path) -> list[Segment]:
    doc = pymupdf.open(str(pdf_path))
    chapter_titles = _chapter_titles(doc)

    segments: dict[tuple, Segment] = {}
    state = {"chapter": "Front matter", "section": None, "subsection": None, "skip": False}
    box_state = {"box": None, "subsection": None}

    def emit(page_no, text, kind="text", label=None, in_box=False):
        # Front matter outside a titled section is the cover/copyright page.
        if state["skip"] or not text or (state["chapter"] == "Front matter" and not state["section"]):
            return
        if in_box and box_state["box"]:
            key_parts = (state["chapter"], None, box_state["subsection"], box_state["box"])
        else:
            key_parts = (state["chapter"], state["section"], state["subsection"], None)
        seg = Segment(*key_parts, kind=kind, label=label)
        seg = segments.setdefault(seg.key, seg)
        seg.pieces.append(Piece(page_no, text))

    def set_chapter(chapter):
        if chapter != state["chapter"]:
            state.update(chapter=chapter, section=None, subsection=None, skip=False)
            box_state.update(box=None, subsection=None)

    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        mid = page.rect.width / 2
        blocks = _blocks(page)
        kinds = [(b, _classify(b, page.rect.height)) for b in blocks]

        # Chapter context comes from the running header (or the opener page).
        for b, k in kinds:
            m = HEADER_RE.match(b.text) if k in ("header", "chapter_title") else None
            if m:
                num = m.group(2)
                set_chapter(f"Chapter {num}: {chapter_titles.get(num, '')}".strip(": ") if num else "Statistical Appendix")
                break

        labels = [b for b, k in kinds if k == "label"]
        page_box = next((b for b in labels if b.text.startswith("BOX")), None)
        if page_box:
            m = LABEL_RE.match(page_box.text)
            title = re.sub(r"\(\s*continued\s*\)", "", m.group(3)).strip()
            box = f"Box {m.group(2)}: {title}"
            if box != box_state["box"]:
                box_state.update(box=box, subsection=None)
        else:
            box_state.update(box=None, subsection=None)

        # Reading order for the text flow: left column top-to-bottom, then right.
        flow = sorted(
            [(b, k) for b, k in kinds if k in ("front_heading", "h2", "h3", "body")],
            key=lambda bk: (bk[0].x0 >= mid - 10, bk[0].y0),
        )
        for i, (b, k) in enumerate(flow):
            text = normalize(b.text)
            if k == "front_heading":
                set_chapter("Front matter")
                state.update(section=text, subsection=None, skip=text in SKIP_FRONT_SECTIONS)
                continue
            if k in ("h2", "h3"):
                # A heading belongs to whichever flow (box or main) its next
                # paragraph belongs to. A heading at the foot of a box page
                # continues on the next box page.
                nxt = next((nb for nb, nk in flow[i + 1 :] if nk == "body"), None)
                if page_box and (nxt is None or nxt.size < 10.9):
                    box_state["subsection"] = text
                elif k == "h2":
                    state.update(section=text, subsection=None, skip=bool(SKIP_SECTIONS_RE.match(text)))
                else:
                    skip = bool(SKIP_SECTIONS_RE.match(text) or SKIP_SECTIONS_RE.match(state["section"] or ""))
                    state.update(subsection=text, skip=skip)
                continue
            emit(page_no, text, in_box=bool(page_box) and b.size < 10.9)

        for b, k in kinds:
            if k == "footnote":
                emit(page_no, normalize(b.text))

        # Figures and tables: group the loose chart/table text under its label.
        graphics = [b for b, k in kinds if k == "graphic"]
        fig_labels = [b for b in labels if not b.text.startswith("BOX")]
        if not fig_labels and graphics and state["chapter"] == "Statistical Appendix":
            # Appendix tables carry a plain title instead of a TABLE label.
            title = min(graphics, key=lambda g: g.y0)
            fig_labels = [title]
            graphics = [g for g in graphics if g is not title]
        if not fig_labels:
            continue
        for lb in fig_labels:
            lb_text = normalize(lb.text)
            is_table = lb_text.startswith("TABLE") or state["chapter"] == "Statistical Appendix"
            group = _assign_graphics(fig_labels, graphics, mid)[id(lb)]
            label = re.sub(r"\(\s*continued\s*\)", "", lb_text).strip()
            emit(
                page_no,
                _graphic_text(lb_text, group, is_table),
                kind="table" if is_table else "figure",
                label=label,
                in_box=bool(page_box) and label.startswith(("FIGURE B", "TABLE B")),
            )

    return list(segments.values())


def main():
    pdf_path = find_pdf()
    segments = parse_pdf(pdf_path)
    by_kind = {}
    for seg in segments:
        by_kind[seg.kind] = by_kind.get(seg.kind, 0) + 1
        chars = sum(len(p.text) for p in seg.pieces)
        pages = seg.pages
        span = f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"
        print(f"{seg.kind:<6} {span:<10} {chars:>6}  {seg.breadcrumb[:120]}")
    print(f"\n{len(segments)} segments: {by_kind}")


if __name__ == "__main__":
    main()
