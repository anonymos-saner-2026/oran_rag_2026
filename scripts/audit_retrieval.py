from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from oran_rag.rag.answerer import RAGEngine


# -----------------------
# OranBench loader
# -----------------------

def load_oranbench(path: str) -> List[Dict[str, Any]]:
    raw = open(path, "r", encoding="utf-8").read().strip()
    if not raw:
        return []

    data: List[Any]
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            data = obj
        else:
            raise ValueError("Top-level JSON is not a list")
    except Exception:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))

    out: List[Dict[str, Any]] = []
    for ex in data:
        if isinstance(ex, list) and len(ex) >= 3:
            q = str(ex[0]).strip()
            opts = ex[1] if isinstance(ex[1], list) else [str(ex[1])]
            gold = str(ex[2]).strip()
            out.append({"question": q, "options": [str(o).strip() for o in opts], "answer": gold})
        elif isinstance(ex, dict):
            q = str(ex.get("question", "")).strip()
            opts = ex.get("options", ex.get("choices", []))
            if not isinstance(opts, list):
                opts = [str(opts)]
            gold = str(ex.get("answer", ex.get("label", ""))).strip()
            out.append({"question": q, "options": [str(o).strip() for o in opts], "answer": gold})
    return out


# -----------------------
# Docstore access
# -----------------------

def open_docstore(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def fetch_chunk_meta(con: sqlite3.Connection, chunk_id: str) -> Dict[str, Any]:
    cur = con.cursor()
    cur.execute(
        "SELECT chunk_id, doc_id, clause_id, title, page_start, page_end, text "
        "FROM chunks WHERE chunk_id=?",
        (chunk_id,),
    )
    row = cur.fetchone()
    if not row:
        return {"chunk_id": chunk_id, "missing": True}
    text = row["text"] or ""
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "clause_id": row["clause_id"],
        "title": row["title"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
        "snippet": text[:700].replace("\n", " ").strip(),
        "missing": False,
    }


# -----------------------
# Simple heuristics: "is this chunk relevant?"
# (Không có ground-truth citations, nên chỉ có proxy score)
# -----------------------

STOP = set("the a an is are was were to of in on for and or with from by between within about as".split())
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/\.]*")

def keywords(text: str) -> List[str]:
    toks = [m.group(0).lower() for m in WORD_RE.finditer(text or "")]
    toks = [t for t in toks if t not in STOP and len(t) >= 3]
    return toks

def overlap_ratio(q: str, chunk_snippet: str) -> float:
    qk = set(keywords(q))
    if not qk:
        return 0.0
    ck = set(keywords(chunk_snippet))
    return len(qk & ck) / max(1, len(qk))


# -----------------------
# (Optional) simulate rewrite that can break retrieval
# -----------------------

def maybe_rewrite_query(engine: RAGEngine, question: str) -> Tuple[str, Dict[str, Any]]:
    """
    Uses engine.rewrite_query (LLM call) -> not desired for retrieval audit by default.
    This function is here ONLY if you pass --rewrite_eval and accept LLM usage.
    """
    # This calls model => slow and can confound retrieval-only test.
    newq = engine.rewrite_query(question)
    return newq, {"rewritten": newq}


# -----------------------
# HTML report
# -----------------------

def write_html(out_path: str, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'/>")
    parts.append("<title>Retrieval Audit Report</title>")
    parts.append("""
<style>
body{font-family:Arial,sans-serif;margin:24px;}
code,pre{background:#f6f8fa;padding:8px;border-radius:8px;white-space:pre-wrap;}
.card{border:1px solid #ddd;border-radius:12px;padding:14px;margin:12px 0;}
table{border-collapse:collapse;width:100%;}
th,td{border-bottom:1px solid #eee;padding:6px 8px;text-align:left;vertical-align:top;}
.small{color:#555;font-size:12px;}
</style></head><body>
""")
    parts.append("<h1>Retrieval Audit Report</h1>")

    parts.append("<div class='card'>")
    parts.append("<h2>Summary</h2>")
    parts.append("<pre>" + esc(json.dumps(summary, ensure_ascii=False, indent=2)) + "</pre>")
    parts.append("</div>")

    for r in rows:
        parts.append("<div class='card'>")
        parts.append(f"<h2>#{r['i']} {esc(r['question'][:120])}</h2>")
        parts.append("<div class='small'>"
                     f"split={esc(r.get('split',''))} | fused_k={r['fused_k']} | rerank_k={r['rerank_k']}"
                     + "</div>")
        parts.append("<h3>Question</h3>")
        parts.append("<pre>" + esc(r["question"]) + "</pre>")
        if r.get("options"):
            parts.append("<h3>Options</h3>")
            parts.append("<pre>" + esc("\n".join(r["options"])) + "</pre>")

        if r.get("rewrite"):
            parts.append("<h3>Rewrite</h3>")
            parts.append("<pre>" + esc(json.dumps(r["rewrite"], ensure_ascii=False, indent=2)) + "</pre>")

        parts.append("<h3>Top Fused</h3>")
        parts.append("<table><tr><th>rank</th><th>score</th><th>doc/clause</th><th>snippet</th><th>chunk_id</th></tr>")
        for j, it in enumerate(r["fused"], start=1):
            parts.append("<tr>"
                         f"<td>{j}</td>"
                         f"<td>{it.get('score')}</td>"
                         f"<td>{esc(it.get('doc_id',''))} | {esc(it.get('clause_id',''))}<br/><span class='small'>{esc(it.get('title',''))}</span></td>"
                         f"<td><pre>{esc(it.get('snippet',''))}</pre></td>"
                         f"<td><code>{esc(it.get('chunk_id',''))}</code></td>"
                         "</tr>")
        parts.append("</table>")

        parts.append("<h3>Top Reranked</h3>")
        parts.append("<table><tr><th>rank</th><th>score</th><th>doc/clause</th><th>snippet</th><th>chunk_id</th></tr>")
        for j, it in enumerate(r["reranked"], start=1):
            parts.append("<tr>"
                         f"<td>{j}</td>"
                         f"<td>{it.get('score')}</td>"
                         f"<td>{esc(it.get('doc_id',''))} | {esc(it.get('clause_id',''))}<br/><span class='small'>{esc(it.get('title',''))}</span></td>"
                         f"<td><pre>{esc(it.get('snippet',''))}</pre></td>"
                         f"<td><code>{esc(it.get('chunk_id',''))}</code></td>"
                         "</tr>")
        parts.append("</table>")

        parts.append("</div>")

    parts.append("</body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# -----------------------
# Main audit
# -----------------------

@dataclass
class ScoreStats:
    fused_scores: List[float]
    rerank_scores: List[float]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval", default=None, help="fin_E.json or fin_M.json or fin_D.json")
    ap.add_argument("--bench_dir", default=None)
    ap.add_argument("--splits", default="E,M,D")
    ap.add_argument("--top_fused", type=int, default=20)
    ap.add_argument("--top_rerank", type=int, default=12)
    ap.add_argument("--max_q", type=int, default=0, help="limit questions (0=all)")
    ap.add_argument("--out_dir", default="data/reports/retrieval_audit")
    ap.add_argument("--rewrite_eval", action="store_true",
                    help="Simulate rewrite (WILL CALL LLM). Default: off.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    engine = RAGEngine.from_config(args.config)
    con = open_docstore(engine.cfg["paths"]["docstore_path"])

    # Determine splits
    split_paths: List[Tuple[str, str]] = []
    if args.eval:
        split_paths.append((os.path.basename(args.eval), args.eval))
    else:
        if not args.bench_dir:
            raise SystemExit("Provide --eval or --bench_dir")
        name_map = {"E": "fin_E.json", "M": "fin_M.json", "D": "fin_D.json"}
        for s in [x.strip().upper() for x in args.splits.split(",") if x.strip()]:
            split_paths.append((s, os.path.join(args.bench_dir, name_map[s])))

    rows_jsonl = os.path.join(args.out_dir, "audit_rows.jsonl")
    html_path = os.path.join(args.out_dir, "audit_report.html")

    # reset jsonl
    open(rows_jsonl, "w", encoding="utf-8").close()

    all_rows: List[Dict[str, Any]] = []
    fused_overlap_scores: List[float] = []
    rerank_overlap_scores: List[float] = []
    fused_score_list: List[float] = []
    rerank_score_list: List[float] = []
    unknown_clause = 0
    total_chunks_seen = 0

    q_counter = 0

    for split_name, path in split_paths:
        data = load_oranbench(path)
        if args.max_q and len(data) > args.max_q:
            data = data[:args.max_q]

        for ex in data:
            q_counter += 1
            q = ex["question"]
            options = ex.get("options", [])
            rewrite_info = None

            q_for_retrieval = q
            if args.rewrite_eval:
                # this will call LLM, so keep optional
                q_for_retrieval, rewrite_info = maybe_rewrite_query(engine, q)

            fused = engine.retrieve(q_for_retrieval, filters=None)
            fused = fused[: args.top_fused]

            reranked = engine.rerank(q_for_retrieval, fused)
            reranked = reranked[: args.top_rerank]

            fused_items: List[Dict[str, Any]] = []
            for score, ch in fused:
                meta = fetch_chunk_meta(con, ch["chunk_id"])
                meta["score"] = float(score)
                fused_items.append(meta)
                total_chunks_seen += 1
                if meta.get("clause_id") == "UNKNOWN":
                    unknown_clause += 1
                fused_score_list.append(float(score))
                fused_overlap_scores.append(overlap_ratio(q, meta.get("snippet", "")))

            rerank_items: List[Dict[str, Any]] = []
            for score, ch in reranked:
                meta = fetch_chunk_meta(con, ch["chunk_id"])
                meta["score"] = float(score)
                rerank_items.append(meta)
                rerank_score_list.append(float(score))
                rerank_overlap_scores.append(overlap_ratio(q, meta.get("snippet", "")))

            row = {
                "i": q_counter,
                "split": split_name,
                "question": q,
                "options": options,
                "q_for_retrieval": q_for_retrieval,
                "rewrite": rewrite_info,
                "fused_k": len(fused_items),
                "rerank_k": len(rerank_items),
                "fused": fused_items,
                "reranked": rerank_items,
            }

            all_rows.append(row)
            with open(rows_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            # light console progress
            if q_counter % 25 == 0:
                print(f"[{q_counter}] split={split_name} done", flush=True)

    def safe_mean(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    def safe_med(xs: List[float]) -> float:
        return float(statistics.median(xs)) if xs else 0.0

    summary = {
        "questions": len(all_rows),
        "splits": [s for s, _ in split_paths],
        "top_fused": args.top_fused,
        "top_rerank": args.top_rerank,
        "rewrite_eval": bool(args.rewrite_eval),
        "unknown_clause_rate_in_fused": (unknown_clause / max(1, total_chunks_seen)),
        "fused_score": {"mean": safe_mean(fused_score_list), "median": safe_med(fused_score_list)},
        "rerank_score": {"mean": safe_mean(rerank_score_list), "median": safe_med(rerank_score_list)},
        "overlap_ratio": {
            "fused_mean": safe_mean(fused_overlap_scores),
            "rerank_mean": safe_mean(rerank_overlap_scores),
            "fused_median": safe_med(fused_overlap_scores),
            "rerank_median": safe_med(rerank_overlap_scores),
        },
        "outputs": {"jsonl": rows_jsonl, "html": html_path},
    }

    write_html(html_path, all_rows, summary)
    with open(os.path.join(args.out_dir, "audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", rows_jsonl)
    print("[OK] wrote:", html_path)
    print("[OK] wrote:", os.path.join(args.out_dir, "audit_summary.json"))


if __name__ == "__main__":
    main()
