from __future__ import annotations
import sqlite3
from typing import Any, Dict, List, Tuple

def fetch_neighbors(
    db_path: str,
    seed_chunks: List[Dict[str, Any]],
    neighbor_window: int = 1
) -> List[Dict[str, Any]]:

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    out_map: Dict[str, Dict[str, Any]] = {}

    for c in seed_chunks:
        section = c.get("section_path", "")
        if not section:
            out_map[c["chunk_id"]] = c
            continue

        # fetch all chunks in same section
        rows = cur.execute(
            '''
            SELECT chunk_id, doc_id, version, wg, clause_id, section_path, title, text, page_start, page_end, is_definition, is_normative
            FROM chunks
            WHERE section_path = ?
            ORDER BY chunk_id ASC
            ''',
            (section,),
        ).fetchall()

        # find position
        ids = [r[0] for r in rows]
        try:
            i = ids.index(c["chunk_id"])
        except ValueError:
            i = None

        if i is None:
            out_map[c["chunk_id"]] = c
            continue

        lo = max(0, i - neighbor_window)
        hi = min(len(rows), i + neighbor_window + 1)
        for r in rows[lo:hi]:
            cc = {
                "chunk_id": r[0],
                "doc_id": r[1],
                "version": r[2],
                "wg": r[3],
                "clause_id": r[4],
                "section_path": r[5],
                "title": r[6],
                "text": r[7],
                "page_start": r[8],
                "page_end": r[9],
                "is_definition": int(r[10]),
                "is_normative": int(r[11]),
            }
            out_map[cc["chunk_id"]] = cc

    con.close()
    out = list(out_map.values())
    # stable order by doc_id then clause_id then chunk_id
    out.sort(key=lambda x: (x.get("doc_id",""), x.get("clause_id",""), x.get("chunk_id","")))
    return out

def trim_contexts_by_chars(contexts: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    total = 0
    for c in contexts:
        t = c.get("text", "")
        if not t:
            continue
        if total + len(t) > max_chars and out:
            break
        out.append(c)
        total += len(t)
    return out

def to_prompt_context(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in contexts:
        out.append(
            {
                "doc_id": c.get("doc_id",""),
                "version": c.get("version",""),
                "wg": c.get("wg",""),
                "clause_id": c.get("clause_id",""),
                "section_path": c.get("section_path",""),
                "pages": f"p{c.get('page_start','?')}-p{c.get('page_end','?')}",
                "text": c.get("text",""),
            }
        )
    return out
