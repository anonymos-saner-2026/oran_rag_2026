from __future__ import annotations

import argparse
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from ..utils import (
    guess_version_from_filename,
    guess_wg_from_filename,
    is_normative_text,
    load_yaml,
    save_jsonl,
)
from ..utils import is_definition_section  # keep if you have it; otherwise adjust import
from .pdf_to_sections import (
    compile_heading_regex,
    detect_headings,
    extract_pdf_text,
    split_into_lines,
)
from .chunker import build_chunks_from_lines


# ----------------------------
# SQLite docstore
# ----------------------------

def ensure_docstore(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            version TEXT,
            wg TEXT,
            clause_id TEXT,
            section_path TEXT,
            title TEXT,
            text TEXT,
            page_start INTEGER,
            page_end INTEGER,
            is_definition INTEGER,
            is_normative INTEGER
        );
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_wg ON chunks(wg);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_clause ON chunks(clause_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_path);")

    con.commit()
    con.close()


def upsert_chunks(db_path: str, rows: List[Dict[str, Any]]) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO chunks
        (chunk_id, doc_id, version, wg, clause_id, section_path, title, text, page_start, page_end, is_definition, is_normative)
        VALUES
        (:chunk_id, :doc_id, :version, :wg, :clause_id, :section_path, :title, :text, :page_start, :page_end, :is_definition, :is_normative)
        """,
        rows,
    )
    con.commit()
    con.close()


# ----------------------------
# Corpus building
# ----------------------------

TOC_DOTLEADER_RE = re.compile(r"\.{3,}\s*(?:\.\s*)*\d+\s*$")



def is_toc_entry_line(page_num: int, ln: str) -> bool:
    # TOC usually early pages; keep conservative
    if page_num > 30:
        return False
    return bool(TOC_DOTLEADER_RE.search(ln or ""))

def build_for_pdf(pdf_path: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    ingest_cfg = cfg["ingest"]
    max_pages = ingest_cfg.get("max_pages_per_pdf", None)

    pages = extract_pdf_text(pdf_path, max_pages=max_pages)
    lines = split_into_lines(pages)

    # Filter TOC dot-leader lines early (avoid noisy tiny chunks)
    # Handles variants like: "Title .......... 12" and "Title .......... .. 12"
    toc_dotleader_re = re.compile(r"\.{3,}\s*(?:\.\s*)*\d+\s*$")

    def is_toc_entry_line(page_num: int, ln: str) -> bool:
        # TOC is usually early pages; keep conservative to not drop real content
        if page_num > 30:
            return False
        return bool(toc_dotleader_re.search(ln or ""))

    lines = [(p, ln) for (p, ln) in lines if not is_toc_entry_line(p, ln)]

    heading_re = compile_heading_regex(ingest_cfg["heading_regex"])
    headings = detect_headings(lines, heading_re)

    file_name = os.path.basename(pdf_path)
    doc_id = os.path.splitext(file_name)[0]
    wg = guess_wg_from_filename(file_name)
    version = guess_version_from_filename(file_name)

    chunks = build_chunks_from_lines(
        doc_id=doc_id,
        version=version,
        wg=wg,
        lines=lines,
        headings=headings,
        min_chars=int(ingest_cfg["min_chunk_chars"]),
        max_chars=int(ingest_cfg["max_chunk_chars"]),
        is_definition_fn=is_definition_section,
        is_normative_fn=is_normative_text,
    )

    # Optional: remove common boilerplate/disclaimer chunks from early pages
    # (helps reduce dup_text noise in retrieval)
    boilerplate_markers = [
        "portal.etsi.org/people/commiteesupport",
        "coordinated vulnerability disclosure",
        "notice of disclaimer",
        "limitation of liability",
        "intellectual property rights",
        "essential patents",
        "publicly available specification",
        "technical specification",
        "ts-family covering the a1 interface",
        "modal verbs terminology",
        "verbal forms for the expression",
        "no guarantee can be given",
        "the present document may be made available in electronic versions",
    ]

    out: List[Dict[str, Any]] = []
    for c in chunks:
        txt = c.text or ""
        low = txt.lower()

        # Drop boilerplate only if it's UNKNOWN (front matter) and very early pages
        if c.clause_id == "UNKNOWN" and c.page_end <= 8:
            strong = any(m in low for m in boilerplate_markers)
            toc_like = "contents" in low and "intellectual property" in low
            if strong or toc_like:
                continue

        out.append(
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "version": c.version,
                "wg": c.wg,
                "clause_id": c.clause_id,
                "section_path": c.section_path,
                "title": c.title,
                "text": txt,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "is_definition": int(bool(c.is_definition)),
                "is_normative": int(bool(c.is_normative)),
            }
        )

    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--reset_docstore", action="store_true", help="Delete docstore.sqlite before building")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    raw_dir = cfg["paths"]["raw_specs_dir"]
    chunks_path = cfg["paths"]["chunks_path"]
    docstore_path = cfg["paths"]["docstore_path"]

    # Optional reset to avoid stale overwrites when chunk_id logic changes
    if args.reset_docstore and os.path.exists(docstore_path):
        os.remove(docstore_path)

    ensure_docstore(docstore_path)

    pdfs = sorted(str(p) for p in Path(raw_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"[ERROR] No PDFs found in {raw_dir}")

    all_rows: List[Dict[str, Any]] = []
    for pdf in pdfs:
        rows = build_for_pdf(pdf, cfg)
        all_rows.extend(rows)
        print(f"[OK] {os.path.basename(pdf)} -> {len(rows)} chunks")

    # Save flat jsonl + upsert into sqlite
    save_jsonl(chunks_path, all_rows)
    upsert_chunks(docstore_path, all_rows)

    print(f"[DONE] chunks.jsonl: {chunks_path} ({len(all_rows)} rows)")
    print(f"[DONE] docstore.sqlite: {docstore_path}")


if __name__ == "__main__":
    main()
