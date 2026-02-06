from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def text_hash(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()

@dataclass
class Stats:
    n: int
    avg: float
    med: float
    p10: int
    p90: int
    minv: int
    maxv: int

def percentile(values, p: float) -> int:
    if not values:
        return 0
    xs = sorted(values)
    k = int(round((len(xs) - 1) * p))
    return int(xs[max(0, min(len(xs) - 1, k))])

def summarize(values) -> Stats:
    if not values:
        return Stats(0, 0, 0, 0, 0, 0, 0)
    xs = list(values)
    return Stats(
        n=len(xs),
        avg=float(mean(xs)),
        med=float(median(xs)),
        p10=percentile(xs, 0.10),
        p90=percentile(xs, 0.90),
        minv=int(min(xs)),
        maxv=int(max(xs)),
    )

def maybe_load_docstore(db_path: str):
    if not db_path or not Path(db_path).exists():
        return None
    con = sqlite3.connect(db_path)
    return con

def fetch_docstore_counts(con: sqlite3.Connection):
    cur = con.cursor()
    n = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    docs = cur.execute("SELECT doc_id, COUNT(*) FROM chunks GROUP BY doc_id ORDER BY COUNT(*) DESC").fetchall()
    return n, docs

def build_report(chunks, out_html: str, out_samples: str, docstore_path: str | None = None):
    # basic counters
    n = len(chunks)
    by_doc = Counter(c.get("doc_id", "UNKNOWN") for c in chunks)
    by_wg = Counter(c.get("wg", "UNKNOWN") for c in chunks)
    by_ver = Counter(c.get("version", "UNKNOWN") for c in chunks)
    by_clause = Counter(c.get("clause_id", "UNKNOWN") for c in chunks)

    # lengths
    lens = [len(c.get("text", "") or "") for c in chunks]
    lines = [(c.get("text", "") or "").count("\n") + 1 for c in chunks]
    len_stats = summarize(lens)
    line_stats = summarize(lines)

    # quality flags
    empty = []
    too_short = []
    too_long = []
    unknown_clause = []
    dup_id = []
    dup_text = []

    seen_ids = set()
    seen_text = defaultdict(list)

    for i, c in enumerate(chunks):
        cid = c.get("chunk_id", f"NO_ID_{i}")
        txt = c.get("text", "") or ""
        if cid in seen_ids:
            dup_id.append(c)
        else:
            seen_ids.add(cid)

        h = text_hash(txt.strip())
        if txt.strip():
            seen_text[h].append(c)
        else:
            empty.append(c)

        L = len(txt)
        if L < 200:
            too_short.append(c)
        if L > 4000:
            too_long.append(c)

        if (c.get("clause_id", "") or "").upper() == "UNKNOWN":
            unknown_clause.append(c)

    for h, lst in seen_text.items():
        if len(lst) >= 3:  # repeated >=3 is suspicious
            dup_text.extend(lst[: min(len(lst), 10)])

    # semantic tags
    n_def = sum(1 for c in chunks if safe_int(c.get("is_definition", 0)) == 1)
    n_norm = sum(1 for c in chunks if safe_int(c.get("is_normative", 0)) == 1)

    # clause explosion: clauses with too many chunks
    clause_exploded = [(cl, ct) for cl, ct in by_clause.items() if ct >= 40 and cl != "UNKNOWN"]
    clause_exploded.sort(key=lambda x: x[1], reverse=True)

    # pick samples for user inspection
    def pick_samples(lst, k=20):
        out = []
        for c in lst[:k]:
            out.append({
                "chunk_id": c.get("chunk_id",""),
                "doc_id": c.get("doc_id",""),
                "version": c.get("version",""),
                "wg": c.get("wg",""),
                "clause_id": c.get("clause_id",""),
                "section_path": c.get("section_path",""),
                "pages": f"p{c.get('page_start','?')}-p{c.get('page_end','?')}",
                "is_definition": safe_int(c.get("is_definition",0)),
                "is_normative": safe_int(c.get("is_normative",0)),
                "text_preview": (c.get("text","") or "")[:800],
            })
        return out

    samples = {
        "empty": pick_samples(empty),
        "too_short": pick_samples(too_short),
        "too_long": pick_samples(too_long),
        "unknown_clause": pick_samples(unknown_clause),
        "dup_id": pick_samples(dup_id),
        "dup_text": pick_samples(dup_text),
        "clause_exploded": clause_exploded[:30],
        "top_docs": by_doc.most_common(20),
        "top_wg": by_wg.most_common(20),
        "top_version": by_ver.most_common(20),
        "top_clause": by_clause.most_common(30),
    }

    Path(out_samples).parent.mkdir(parents=True, exist_ok=True)
    with open(out_samples, "w", encoding="utf-8") as f:
        for key, items in samples.items():
            f.write(json.dumps({"bucket": key, "items": items}, ensure_ascii=False) + "\n")

    # optional docstore cross-check
    docstore_info = ""
    con = None
    if docstore_path:
        con = maybe_load_docstore(docstore_path)
    if con is not None:
        try:
            n_db, docs_db = fetch_docstore_counts(con)
            docstore_info = f"<p><b>Docstore:</b> {docstore_path} | rows={n_db}</p>"
        finally:
            con.close()

    # HTML report
    def fmt_stats(s: Stats) -> str:
        return f"n={s.n}, avg={s.avg:.1f}, med={s.med:.1f}, p10={s.p10}, p90={s.p90}, min={s.minv}, max={s.maxv}"

    def pct(x: int) -> str:
        return f"{(100.0 * x / max(1,n)):.2f}%"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Chunk Quality Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    code, pre {{ background: #f6f8fa; padding: 8px; border-radius: 8px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 14px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; vertical-align: top; }}
    .bad {{ color: #b00020; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>Chunk Quality Report</h1>
  <p><b>Total chunks:</b> {n}</p>
  {docstore_info}

  <div class="grid">
    <div class="card">
      <h2>Length stats</h2>
      <p><b>Chars:</b> {fmt_stats(len_stats)}</p>
      <p><b>Lines:</b> {fmt_stats(line_stats)}</p>
    </div>
    <div class="card">
      <h2>Flags</h2>
      <p><span class="bad">Empty:</span> {len(empty)} ({pct(len(empty))})</p>
      <p><span class="bad">Too short (&lt;200 chars):</span> {len(too_short)} ({pct(len(too_short))})</p>
      <p><span class="bad">Too long (&gt;4000 chars):</span> {len(too_long)} ({pct(len(too_long))})</p>
      <p><span class="bad">Unknown clause_id:</span> {len(unknown_clause)} ({pct(len(unknown_clause))})</p>
      <p><span class="bad">Duplicate chunk_id:</span> {len(dup_id)} ({pct(len(dup_id))})</p>
      <p><span class="bad">Repeated text (hash>=3):</span> {len(dup_text)} ({pct(len(dup_text))})</p>
      <p><b>is_definition:</b> {n_def} ({pct(n_def)})</p>
      <p><b>is_normative:</b> {n_norm} ({pct(n_norm)})</p>
    </div>
  </div>

  <div class="card">
    <h2>Top distributions</h2>
    <div class="grid">
      <div>
        <h3>Top doc_id</h3>
        <table>
          <tr><th>doc_id</th><th>count</th></tr>
          {''.join([f"<tr><td>{d}</td><td>{c}</td></tr>" for d,c in by_doc.most_common(15)])}
        </table>
      </div>
      <div>
        <h3>Top clause_id</h3>
        <table>
          <tr><th>clause_id</th><th>count</th></tr>
          {''.join([f"<tr><td>{cl}</td><td>{c}</td></tr>" for cl,c in by_clause.most_common(20)])}
        </table>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Clauses with too many chunks (>=40)</h2>
    <pre>{json.dumps(clause_exploded[:50], ensure_ascii=False, indent=2)}</pre>
  </div>

  <div class="card">
    <h2>Samples (see JSONL)</h2>
    <p>Samples written to: <code>{out_samples}</code></p>
    <p>Open the file and search by bucket: <code>empty</code>, <code>too_short</code>, <code>unknown_clause</code>, <code>dup_text</code>...</p>
  </div>

</body>
</html>
"""
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default="data/intermediate/chunks.jsonl")
    ap.add_argument("--docstore", type=str, default="data/indexes/docstore.sqlite")
    ap.add_argument("--out_html", type=str, default="data/reports/chunk_report.html")
    ap.add_argument("--out_samples", type=str, default="data/reports/chunk_samples.jsonl")
    args = ap.parse_args()

    if not Path(args.chunks).exists():
        raise SystemExit(f"[ERROR] chunks file not found: {args.chunks}")

    chunks = read_jsonl(args.chunks)
    build_report(chunks, args.out_html, args.out_samples, docstore_path=args.docstore)

    print("[DONE] HTML report:", args.out_html)
    print("[DONE] Samples JSONL:", args.out_samples)
    print("Tip: open the HTML file in your browser.")

if __name__ == "__main__":
    main()
