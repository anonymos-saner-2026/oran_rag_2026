from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PyPDF2 import PdfReader


@dataclass
class PageText:
    page_num: int
    text: str


def extract_pdf_text(pdf_path: str, max_pages: int | None = None) -> List[PageText]:
    reader = PdfReader(pdf_path)
    n = len(reader.pages)
    if max_pages is not None:
        n = min(n, max_pages)

    pages: List[PageText] = []
    for i in range(n):
        page = reader.pages[i]
        txt = page.extract_text() or ""
        # Normalize NBSP; keep line breaks (caller will split)
        txt = txt.replace("\u00A0", " ")
        pages.append(PageText(page_num=i + 1, text=txt))
    return pages


def split_into_lines(pages: List[PageText]) -> List[Tuple[int, str]]:
    lines: List[Tuple[int, str]] = []
    for p in pages:
        for ln in (p.text or "").splitlines():
            ln = ln.rstrip()
            if ln.strip():
                lines.append((p.page_num, ln))
    return lines


def compile_heading_regex(pattern: str) -> re.Pattern:
    return re.compile(pattern, flags=re.IGNORECASE)


@dataclass
class Heading:
    number: str
    title: str
    page_num: int


# --------- Heuristics to reduce false headings (TOC + table IDs + dates) ---------

_TOC_DOTLEADER_RE = re.compile(r"\.{3,}\s*(?:\.\s*)*\d+\s*$")



def is_toc_line(s: str) -> bool:
    # Table of contents dot leaders: "Title ....... 27"
    return bool(_TOC_DOTLEADER_RE.search(s or ""))


def clean_toc_suffix(title: str) -> str:
    # remove trailing dot-leader page numbers if present
    return _TOC_DOTLEADER_RE.sub("", title or "").strip()


def plausible_clause_number(num: str) -> bool:
    """
    Reject obvious non-clause identifiers:
    - "28.552" (1 dot but 3+ digits in last segment)
    - date-like "YYYY.MM.DD" (e.g., 2022.11.17)
    - absurdly large first segment
    ETSI/O-RAN clause numbering is dot-separated, usually 1-2 digits per segment.
    """
    parts = (num or "").split(".")
    if not parts or not all(p.isdigit() for p in parts):
        return False

    # reject date-like numbering: YYYY.MM.DD
    if len(parts) == 3 and len(parts[0]) == 4 and (1900 <= int(parts[0]) <= 2100):
        mm = int(parts[1])
        dd = int(parts[2])
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return False

    # safety: reject very large first segment (also blocks YYYY.* even if not caught above)
    if int(parts[0]) > 2000:
        return False

    # reject "28.552" pattern (2 segments, last one has 3+ digits)
    if len(parts) == 2 and len(parts[1]) >= 3:
        return False

    # reject any segment after the first being 3+ digits
    if any(len(p) >= 3 for p in parts[1:]):
        return False

    return True


def detect_headings(lines: List[Tuple[int, str]], heading_re: re.Pattern) -> List[Heading]:
    """
    Detect headings based on regex from config, with safeguards:
      - Skip TOC entries like "4.6 ... .... 27"
      - Skip implausible numbers like "28.552" or date-like "2022.11.17"
      - Clean trailing ".... 27" from title
    """
    hs: List[Heading] = []
    for page_num, ln in lines:
        m = heading_re.match(ln)
        if not m:
            continue

        num = (m.group(1) or "").strip()
        title = (m.group(2) or "").strip()

        # TOC guard
        if is_toc_line(ln) or is_toc_line(title):
            continue

        title = clean_toc_suffix(title)

        # clause plausibility guard
        if not plausible_clause_number(num):
            continue

        # allow "7 Scope" etc.
        if len(num) >= 1 and len(title) >= 2:
            hs.append(Heading(number=num, title=title, page_num=page_num))

    return hs


def nearest_heading_for_line_index(
    headings_idx: List[Tuple[int, Heading]],
    line_i: int,
) -> Optional[Heading]:
    # headings_idx sorted by line index
    last: Optional[Heading] = None
    for idx, h in headings_idx:
        if idx > line_i:
            break
        last = h
    return last
