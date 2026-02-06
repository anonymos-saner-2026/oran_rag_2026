#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

P_UTILS = ROOT / "src/oran_rag/utils.py"
P_PDF = ROOT / "src/oran_rag/ingest/pdf_to_sections.py"
P_BUILD = ROOT / "src/oran_rag/ingest/build_corpus.py"
P_CHUNKER = ROOT / "src/oran_rag/ingest/chunker.py"

def backup(p: Path) -> Path:
    b = p.with_suffix(p.suffix + f".bak_{STAMP}")
    shutil.copy2(p, b)
    return b

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write(p: Path, s: str) -> None:
    p.write_text(s if s.endswith("\n") else s + "\n", encoding="utf-8")

def replace_function(text: str, func_name: str, new_func_src: str) -> str:
    # Replace "def func_name(...): ... " until next top-level "def " or EOF
    pat = rf"^def\s+{re.escape(func_name)}\s*\(.*?\):.*?(?=^\s*def\s+|\Z)"
    m = re.search(pat, text, flags=re.DOTALL | re.MULTILINE)
    if not m:
        raise RuntimeError(f"Cannot find function: {func_name}")
    return text[:m.start()] + new_func_src.rstrip() + "\n\n" + text[m.end():].lstrip()

def ensure_after_line(text: str, anchor_line_substr: str, insert_block: str) -> str:
    if insert_block.strip() in text:
        return text
    lines = text.splitlines(True)
    for i, ln in enumerate(lines):
        if anchor_line_substr in ln:
            lines.insert(i + 1, "\n" + insert_block.rstrip() + "\n\n")
            return "".join(lines)
    raise RuntimeError(f"Anchor not found: {anchor_line_substr}")

def patch_utils():
    s = read(P_UTILS)
    if "Prefer pattern with explicit 'v'" in s:
        print("[SKIP] utils.py already patched.")
        return
    new_func = """def guess_version_from_filename(name: str) -> str:
    # Prefer pattern with explicit 'v' (case-insensitive), allow '_' or other non-alnum before v
    m = re.search(r"(?:^|[^0-9A-Za-z])v(\\d+\\.\\d+(?:\\.\\d+)*)", name, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # fallback: first x.y(.z) occurrence with non-alnum boundary
    m = re.search(r"(?:^|[^0-9A-Za-z])(\\d+\\.\\d+(?:\\.\\d+)*)", name)
    if m:
        return m.group(1)
    return "UNKNOWN"
"""
    s2 = replace_function(s, "guess_version_from_filename", new_func)
    write(P_UTILS, s2)
    print("[OK] Patched utils.py (version parsing).")

def patch_pdf_to_sections():
    s = read(P_PDF)

    # Insert helper functions after "from PyPDF2 import PdfReader"
    helpers = """def is_toc_line(s: str) -> bool:
    # Table of contents dot leaders: "Title ....... 27"
    return bool(re.search(r"\\.{3,}\\s*\\d+\\s*$", s or ""))

def plausible_clause_number(num: str) -> bool:
    parts = (num or "").split(".")
    if not parts or not all(p.isdigit() for p in parts):
        return False
    # reject "28.552" (one dot but last segment has 3+ digits)
    if len(parts) == 2 and len(parts[1]) >= 3:
        return False
    # generally, clause segments after the first should be 1-2 digits
    if any(len(p) >= 3 for p in parts[1:]):
        return False
    return True
"""
    if "def is_toc_line" not in s:
        s = ensure_after_line(s, "from PyPDF2 import PdfReader", helpers)
        print("[OK] Inserted helpers into pdf_to_sections.py")
    else:
        print("[SKIP] pdf_to_sections.py helpers already present.")

    # Replace detect_headings entirely
    new_detect = """def detect_headings(lines: List[Tuple[int, str]], heading_re: re.Pattern) -> List[Heading]:
    \"\"\"Detect headings like: 7.3.2 Some title
    Safeguards:
      - skip TOC dot-leader lines
      - skip implausible clause numbers like 28.552
    \"\"\"
    hs: List[Heading] = []
    for page_num, ln in lines:
        m = heading_re.match(ln)
        if not m:
            continue
        num = m.group(1).strip()
        title = (m.group(2) or "").strip()

        if is_toc_line(ln) or is_toc_line(title):
            continue

        # remove trailing dot-leader page numbers if still present
        title = re.sub(r"\\.{3,}\\s*\\d+\\s*$", "", title).strip()

        if not plausible_clause_number(num):
            continue

        if len(num) >= 1 and len(title) >= 2:
            hs.append(Heading(number=num, title=title, page_num=page_num))
    return hs
"""
    s2 = replace_function(s, "detect_headings", new_detect)
    write(P_PDF, s2)
    print("[OK] Patched pdf_to_sections.py detect_headings().")

def patch_build_corpus():
    s = read(P_BUILD)
    if "def is_toc_entry_line" in s:
        print("[SKIP] build_corpus.py already patched.")
        return

    needle = "lines = split_into_lines(pages)"
    idx = s.find(needle)
    if idx < 0:
        raise RuntimeError("Cannot find 'lines = split_into_lines(pages)' in build_corpus.py")

    insert = """    import re
    def is_toc_entry_line(page_num: int, ln: str) -> bool:
        # Be conservative: TOC usually early pages
        if page_num > 30:
            return False
        return bool(re.search(r"\\.{3,}\\s*\\d+\\s*$", ln or ""))

    lines = [(p, ln) for (p, ln) in lines if not is_toc_entry_line(p, ln)]
"""
    # insert right after the needle line (keep indentation)
    lines = s.splitlines(True)
    out = []
    inserted = False
    for ln in lines:
        out.append(ln)
        if not inserted and needle in ln:
            out.append(insert + "\n")
            inserted = True

    if not inserted:
        raise RuntimeError("Failed to insert TOC filter in build_corpus.py")

    write(P_BUILD, "".join(out))
    print("[OK] Patched build_corpus.py (TOC line filter).")

def patch_chunker():
    s = read(P_CHUNKER)

    # 1) Make chunk_id unique by including page range (prevents dup_id + sqlite overwrite).
    # Replace make_chunk_id signature and call site.
    if "def make_chunk_id(" in s and "page_start" not in s.split("def make_chunk_id", 1)[1].split(")", 1)[0]:
        s = re.sub(
            r"def\s+make_chunk_id\s*\(\s*doc_id\s*:\s*str\s*,\s*clause_id\s*:\s*str\s*,\s*idx\s*:\s*int\s*\)\s*->\s*str\s*:\s*\n\s*return\s+f\"{doc_id}::\{clause_id\}::\{idx:04d\}\"",
            "def make_chunk_id(doc_id: str, clause_id: str, page_start: int, page_end: int, idx: int) -> str:\n    return f\"{doc_id}::{clause_id}::p{page_start}-{page_end}::{idx:04d}\"",
            s,
            count=1,
            flags=re.MULTILINE,
        )

    # Replace call site chunk_id=make_chunk_id(doc_id, clause_id, j) -> include pages
    s = s.replace(
        "chunk_id=make_chunk_id(doc_id, clause_id, j),",
        "chunk_id=make_chunk_id(doc_id, clause_id, pages[0], pages[1], j),",
    )

    # 2) Hard split safeguard: prevent huge 27k chunks when no punctuation.
    if "def hard_split" not in s:
        # Insert helper near the split branch inside build_chunks_from_lines
        s = s.replace(
            '# split by ". " or "; " conservatively',
            '# split by ". " or "; " conservatively\n                    def hard_split(s: str, max_chars: int):\n                        return [s[i:i+max_chars] for i in range(0, len(s), max_chars)]\n',
            1
        )

    # Patch the loop "for part in parts:" to handle oversize parts.
    # Replace the first occurrence of:
    # for part in parts:
    #     if not part:
    #         continue
    # with improved version.
    s = re.sub(
        r"for\s+part\s+in\s+parts:\s*\n\s*if\s+not\s+part:\s*\n\s*continue\s*\n",
        "for part in parts:\n                        part = part.strip()\n                        if not part:\n                            continue\n                        if len(part) > max_chars:\n                            if buf:\n                                pieces.append(buf)\n                                buf = \"\"\n                            pieces.extend(hard_split(part, max_chars))\n                            continue\n",
        s,
        count=1,
        flags=re.MULTILINE,
    )

    # 3) Merge last tiny chunk into previous (reduce too_short).
    if "merge last tiny chunk into previous" not in s:
        s = s.replace(
            "if buf:\n                    merged.append(buf)\n",
            "if buf:\n                    merged.append(buf)\n\n                # merge last tiny chunk into previous\n                if len(merged) >= 2 and len(merged[-1]) < min_chars:\n                    merged[-2] = (merged[-2] + \" \" + merged[-1]).strip()\n                    merged.pop()\n",
            1
        )

    write(P_CHUNKER, s)
    print("[OK] Patched chunker.py (unique chunk_id + hard split + merge tail).")

def main():
    for p in [P_UTILS, P_PDF, P_BUILD, P_CHUNKER]:
        if not p.exists():
            raise SystemExit(f"[ERROR] Missing file: {p}")

    # backup only files that haven't been backed up this run
    print("[*] Backing up files...")
    for p in [P_UTILS, P_PDF, P_BUILD, P_CHUNKER]:
        b = backup(p)
        print("   -", p.relative_to(ROOT), "->", b.name)

    print("[*] Applying patches...")
    patch_utils()
    patch_pdf_to_sections()
    patch_build_corpus()
    patch_chunker()

    print("\n[DONE] Patch v1.1 applied.\n")
    print("Rebuild clean artifacts:")
    print("  rm -f data/intermediate/chunks.jsonl")
    print("  rm -f data/indexes/docstore.sqlite")
    print("  rm -rf data/indexes/bm25 data/indexes/faiss")
    print("  PYTHONPATH=src python -m oran_rag.ingest.build_corpus --config configs/app.yaml")
    print("  PYTHONPATH=src python -m oran_rag.eval.chunk_report")

if __name__ == "__main__":
    main()
