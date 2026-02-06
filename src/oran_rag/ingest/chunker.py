from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    version: str
    wg: str
    clause_id: str
    section_path: str
    title: str
    text: str
    page_start: int
    page_end: int
    is_definition: bool
    is_normative: bool


def normalize_text(s: str) -> str:
    s = (s or "").replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def make_chunk_id(doc_id: str, clause_id: str, page_start: int, page_end: int, block_i: int, idx: int, text: str) -> str:
    t = text or ""
    h = hashlib.md5(t.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}::{clause_id}::p{page_start}-{page_end}::b{block_i:04d}::{idx:04d}::{len(t)}::{h}"




def hard_split(s: str, max_chars: int) -> List[str]:
    return [s[i : i + max_chars] for i in range(0, len(s), max_chars)]


def build_section_path(curr: Optional[Tuple[str, str]]) -> str:
    if not curr:
        return "UNKNOWN"
    num, title = curr
    title = (title or "").strip()
    return f"{num} {title}".strip()


def pack_short_chunks(texts: List[str], min_keep: int, max_chars: int) -> List[str]:
    """
    Merge small segments into neighbors (prefer previous) to reduce too_short noise.
    """
    out: List[str] = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue

        if len(t) < min_keep:
            if out and (len(out[-1]) + 1 + len(t) <= max_chars):
                out[-1] = (out[-1] + " " + t).strip()
            else:
                out.append(t)
        else:
            out.append(t)

    # backward merge tiny tail
    if len(out) >= 2 and len(out[-1]) < min_keep and (len(out[-2]) + 1 + len(out[-1]) <= max_chars):
        out[-2] = (out[-2] + " " + out[-1]).strip()
        out.pop()

    return out


def build_chunks_from_lines(
    doc_id: str,
    version: str,
    wg: str,
    lines: List[Tuple[int, str]],
    headings: List,  # List[Heading] from pdf_to_sections
    min_chars: int = 400,
    max_chars: int = 2200,
    is_definition_fn: Optional[Callable[[str], bool]] = None,
    is_normative_fn: Optional[Callable[[str], bool]] = None,
) -> List[Chunk]:
    """
    Strategy:
    - Align headings to line indices using robust token overlap.
    - Assign each line to nearest preceding heading.
    - Group contiguous lines by heading clause number.
    - Chunk each block by sentence-ish splitting + hard split.
    - Pack tiny chunks and optionally drop super tiny junk.
    """

    # Map headings by page
    by_page: Dict[int, List[Tuple[str, str]]] = {}
    for h in headings:
        by_page.setdefault(h.page_num, []).append((h.number, h.title))

    # Align headings to line indices
    headings_idx: List[Tuple[int, Tuple[str, str]]] = []
    for i, (p, ln) in enumerate(lines):
        ln_low = ln.strip().lower()
        if not ln_low:
            continue

        for (num, title) in by_page.get(p, []):
            num_low = (num or "").lower()
            title_low = (title or "").lower()

            if not num_low:
                continue

            # Must start with the clause number
            if not ln_low.startswith(num_low):
                continue

            toks = [w for w in re.findall(r"[a-z0-9]{3,}", title_low)][:6]
            if not toks:
                headings_idx.append((i, (num, title)))
                break

            hit = sum(1 for w in toks if w in ln_low)
            need = 1 if len(toks) <= 2 else 2
            if hit >= need:
                headings_idx.append((i, (num, title)))
                break

    # De-duplicate heading indices by (idx, clause_id)
    dedup = {}
    for idx, ht in headings_idx:
        key = (idx, ht[0])
        if key not in dedup:
            dedup[key] = ht
    headings_idx = sorted([(idx, ht) for (idx, _), ht in dedup.items()], key=lambda x: x[0])

    def nearest_heading(line_i: int) -> Optional[Tuple[str, str]]:
        last = None
        for idx, h in headings_idx:
            if idx > line_i:
                break
            last = h
        return last

    # Group contiguous lines by heading clause_id
    blocks: List[Dict] = []
    curr_key: Optional[str] = None
    curr_lines: List[Tuple[int, str]] = []
    curr_heading: Optional[Tuple[str, str]] = None

    for i, (p, ln) in enumerate(lines):
        h = nearest_heading(i)
        key = (h[0] if h else "UNKNOWN")

        if curr_key is None:
            curr_key = key
            curr_heading = h
            curr_lines = [(p, ln)]
            continue

        if key == curr_key:
            curr_lines.append((p, ln))
        else:
            blocks.append({"heading": curr_heading, "lines": curr_lines})
            curr_key = key
            curr_heading = h
            curr_lines = [(p, ln)]

    if curr_lines:
        blocks.append({"heading": curr_heading, "lines": curr_lines})

    chunks: List[Chunk] = []

    # Drop extremely short junk after packing (but keep some definition lines)
    MIN_INDEX_CHARS = 120
    MIN_DEF_KEEP = 80  # allow shorter definition entries

    for block_i, block in enumerate(blocks):

        heading = block["heading"]
        block_lines: List[Tuple[int, str]] = block["lines"]

        page_start = block_lines[0][0]
        page_end = block_lines[-1][0]

        clause_id = heading[0] if heading else "UNKNOWN"
        title = heading[1] if heading else "UNKNOWN"
        section_path = build_section_path(heading)

        raw_text = "\n".join([ln for _, ln in block_lines])
        raw_text = normalize_text(raw_text)
        if not raw_text:
            continue

        # split by sentence-ish boundaries
        parts = re.split(r"(?<=[\.\?\!;:])\s+", raw_text)

        pieces: List[str] = []
        buf = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part) > max_chars:
                if buf:
                    pieces.append(buf)
                    buf = ""
                pieces.extend(hard_split(part, max_chars))
                continue

            if not buf:
                buf = part
            elif len(buf) + 1 + len(part) <= max_chars:
                buf = f"{buf} {part}".strip()
            else:
                pieces.append(buf)
                buf = part

        if buf:
            pieces.append(buf)

        # forward merge to enforce min_chars
        merged: List[str] = []
        carry = ""
        for piece in pieces:
            if not carry:
                carry = piece
                continue
            if len(carry) < min_chars:
                carry = f"{carry} {piece}".strip()
            else:
                merged.append(carry)
                carry = piece
        if carry:
            merged.append(carry)

        # pack tiny chunks (<200)
        merged = pack_short_chunks(merged, min_keep=200, max_chars=max_chars)

        # final backward merge for tiny tail (<200)
        if len(merged) >= 2 and len(merged[-1]) < 200 and (len(merged[-2]) + 1 + len(merged[-1]) <= max_chars):
            merged[-2] = (merged[-2] + " " + merged[-1]).strip()
            merged.pop()

        # drop super tiny junk (keep some short definition entries)
        filtered: List[str] = []
        for t in merged:
            t2 = t.strip()
            if len(t2) >= MIN_INDEX_CHARS:
                filtered.append(t2)
                continue
            if is_definition_fn and is_definition_fn(t2) and len(t2) >= MIN_DEF_KEEP:
                filtered.append(t2)
                continue
        merged = filtered

        for j, txt in enumerate(merged):
            txt = normalize_text(txt)
            if not txt:
                continue

            # final safety drop (rare)
            if len(txt) < MIN_DEF_KEEP:
                continue

            is_def = bool(is_definition_fn(txt)) if is_definition_fn else False
            is_norm = bool(is_normative_fn(txt)) if is_normative_fn else False

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_id, clause_id, page_start, page_end, block_i, j, txt),


                    doc_id=doc_id,
                    version=version,
                    wg=wg,
                    clause_id=clause_id,
                    section_path=section_path,
                    title=title,
                    text=txt,
                    page_start=page_start,
                    page_end=page_end,
                    is_definition=is_def,
                    is_normative=is_norm,
                )
            )

    return chunks
