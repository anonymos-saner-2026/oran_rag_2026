from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..utils import load_yaml, read_jsonl

try:
    from ..index import bm25_index
except Exception:
    bm25_index = None  # type: ignore

try:
    from ..index import dense_index
except Exception:
    dense_index = None  # type: ignore


@dataclass
class Stats:
    n: int
    avg: float
    med: float
    p10: int
    p90: int
    minv: int
    maxv: int


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def text_hash(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def percentile(values: List[int], p: float) -> int:
    if not values:
        return 0
    xs = sorted(values)
    k = int(round((len(xs) - 1) * p))
    return int(xs[max(0, min(len(xs) - 1, k))])


def summarize(values: Iterable[int]) -> Stats:
    xs = list(values)
    if not xs:
        return Stats(0, 0.0, 0.0, 0, 0, 0, 0)
    return Stats(
        n=len(xs),
        avg=float(mean(xs)),
        med=float(median(xs)),
        p10=percentile(xs, 0.10),
        p90=percentile(xs, 0.90),
        minv=int(min(xs)),
        maxv=int(max(xs)),
    )


def write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows(rows: List[Dict[str, Any]], k: int, seed: int) -> List[Dict[str, Any]]:
    if k <= 0:
        return []
    rng = random.Random(seed)
    if len(rows) <= k:
        return rows
    return rng.sample(rows, k)


def chunk_preview(c: Dict[str, Any], max_chars: int = 500) -> Dict[str, Any]:
    text = (c.get("text", "") or "").strip()
    preview = text[:max_chars]
    return {
        "chunk_id": c.get("chunk_id", ""),
        "doc_id": c.get("doc_id", ""),
        "version": c.get("version", ""),
        "wg": c.get("wg", ""),
        "clause_id": c.get("clause_id", ""),
        "section_path": c.get("section_path", ""),
        "page_start": c.get("page_start", ""),
        "page_end": c.get("page_end", ""),
        "length_chars": len(text),
        "text_preview": preview,
    }


def build_query_from_chunk(c: Dict[str, Any], max_chars: int = 400) -> str:
    title = str(c.get("title", "") or "")
    clause = str(c.get("clause_id", "") or "")
    text = str(c.get("text", "") or "").strip()
    text = text[:max_chars]
    return f"{title}\n{clause}\n{text}".strip()


def fetch_docstore_chunk_ids(db_path: str) -> Tuple[int, List[str]]:
    if not db_path or not Path(db_path).exists():
        return 0, []
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        n = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        rows = cur.execute("SELECT chunk_id FROM chunks").fetchall()
        ids = [r[0] for r in rows if r and r[0]]
        return int(n), ids
    finally:
        con.close()


def self_retrieval_test(
    chunks: List[Dict[str, Any]],
    search_fn,
    topk: int,
    sample_n: int,
    seed: int,
) -> Dict[str, Any]:
    sampled = sample_rows(chunks, sample_n, seed)
    hits = 0
    ranks: List[int] = []
    misses: List[Dict[str, Any]] = []

    for c in sampled:
        query = build_query_from_chunk(c)
        results = search_fn(query, topk)
        found_rank = 0
        for i, (_, r) in enumerate(results, start=1):
            if r.get("chunk_id") == c.get("chunk_id"):
                found_rank = i
                break
        if found_rank > 0:
            hits += 1
            ranks.append(found_rank)
        else:
            misses.append(
                {
                    "chunk_id": c.get("chunk_id", ""),
                    "query_preview": query[:300],
                    "top_chunk_ids": [r.get("chunk_id", "") for _, r in results[:5]],
                }
            )

    hit_rate = hits / max(1, len(sampled))
    mean_rank = float(mean(ranks)) if ranks else 0.0
    return {
        "sample_n": len(sampled),
        "topk": topk,
        "hit_rate": hit_rate,
        "mean_rank": mean_rank,
        "misses": misses,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/app.yaml")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--sample_n", type=int, default=200)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip_dense", action="store_true")
    ap.add_argument("--skip_bm25", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    ingest_cfg = cfg.get("ingest", {})

    out_dir = args.out_dir or os.path.join("data", "reports", f"corpus_index_report_{now_tag()}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    chunks_path = paths.get("chunks_path", "data/intermediate/chunks.jsonl")
    docstore_path = paths.get("docstore_path", "data/indexes/docstore.sqlite")
    bm25_dir = paths.get("bm25_dir", "data/indexes/bm25")
    faiss_dir = paths.get("faiss_dir", "data/indexes/faiss")

    if not Path(chunks_path).exists():
        raise SystemExit(f"[ERROR] chunks file not found: {chunks_path}")

    chunks = read_jsonl(chunks_path)
    chunk_ids = [c.get("chunk_id", "") for c in chunks if c.get("chunk_id")]
    chunk_id_set = set(chunk_ids)

    # corpus stats
    raw_specs_dir = paths.get("raw_specs_dir", "data/raw_specs")
    pdfs = sorted(Path(raw_specs_dir).glob("*.pdf"))
    pdf_sizes = [p.stat().st_size for p in pdfs]

    # chunk stats
    lens = [len((c.get("text", "") or "")) for c in chunks]
    words = [len((c.get("text", "") or "").split()) for c in chunks]
    len_stats = summarize(lens)
    word_stats = summarize(words)

    min_expected = int(ingest_cfg.get("min_chunk_chars", 400))
    max_expected = int(ingest_cfg.get("max_chunk_chars", 2200))

    empty = [c for c in chunks if not (c.get("text", "") or "").strip()]
    too_short = [c for c in chunks if 0 < len((c.get("text", "") or "")) < min_expected]
    too_long = [c for c in chunks if len((c.get("text", "") or "")) > max_expected]
    unknown_clause = [c for c in chunks if str(c.get("clause_id", "")).upper() == "UNKNOWN"]

    # duplicates
    dup_chunk_ids = []
    seen_ids: Dict[str, int] = {}
    for c in chunks:
        cid = c.get("chunk_id", "")
        if not cid:
            continue
        seen_ids[cid] = seen_ids.get(cid, 0) + 1
    dup_chunk_ids = [cid for cid, ct in seen_ids.items() if ct > 1]

    text_hashes: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        txt = (c.get("text", "") or "").strip()
        if not txt:
            continue
        h = text_hash(txt)
        text_hashes.setdefault(h, []).append(c)
    dup_text_groups = [v for v in text_hashes.values() if len(v) >= 3]
    dup_text_samples: List[Dict[str, Any]] = []
    for grp in dup_text_groups[:50]:
        dup_text_samples.extend(grp[:3])

    samples_path = os.path.join(out_dir, "corpus_index_report.samples.jsonl")
    open(samples_path, "w", encoding="utf-8").close()

    for c in sample_rows(empty, 20, args.seed):
        append_jsonl(samples_path, {"bucket": "empty", "item": chunk_preview(c)})
    for c in sample_rows(too_short, 20, args.seed):
        append_jsonl(samples_path, {"bucket": "too_short", "item": chunk_preview(c)})
    for c in sample_rows(too_long, 20, args.seed):
        append_jsonl(samples_path, {"bucket": "too_long", "item": chunk_preview(c)})
    for c in sample_rows(unknown_clause, 20, args.seed):
        append_jsonl(samples_path, {"bucket": "unknown_clause", "item": chunk_preview(c)})
    for cid in dup_chunk_ids[:20]:
        append_jsonl(samples_path, {"bucket": "dup_chunk_id", "item": {"chunk_id": cid}})
    for c in sample_rows(dup_text_samples, 20, args.seed):
        append_jsonl(samples_path, {"bucket": "dup_text", "item": chunk_preview(c)})

    # docstore coverage
    docstore_exists = Path(docstore_path).exists()
    docstore_rows, docstore_ids = fetch_docstore_chunk_ids(docstore_path)
    docstore_id_set = set(docstore_ids)
    if docstore_exists:
        missing_in_docstore = sorted(chunk_id_set - docstore_id_set)
        extra_in_docstore = sorted(docstore_id_set - chunk_id_set)
    else:
        missing_in_docstore = []
        extra_in_docstore = []

    for cid in missing_in_docstore[:20]:
        append_jsonl(samples_path, {"bucket": "missing_in_docstore", "item": {"chunk_id": cid}})
    for cid in extra_in_docstore[:20]:
        append_jsonl(samples_path, {"bucket": "extra_in_docstore", "item": {"chunk_id": cid}})

    # BM25 index checks
    bm25_summary: Dict[str, Any] = {"status": "skipped"}
    if not args.skip_bm25:
        bm25_pkl = os.path.join(bm25_dir, "bm25.pkl")
        if bm25_index is None:
            bm25_summary = {"status": "unavailable", "reason": "bm25_index import failed"}
        elif Path(bm25_pkl).exists():
            bm25_mod: Any = bm25_index
            try:
                idx = bm25_mod.load_index(bm25_dir)
            except Exception as exc:
                bm25_summary = {"status": "error", "error": str(exc), "bm25_pkl": bm25_pkl}
                idx = None
            if idx is None:
                pass
            else:
                bm25_chunks = idx.get("chunks", [])
                bm25_ids = [c.get("chunk_id", "") for c in bm25_chunks if c.get("chunk_id")]
                bm25_id_set = set(bm25_ids)
                def bm25_search(query: str, topk: int) -> List[Tuple[float, Dict[str, Any]]]:
                    return bm25_mod.search(idx, query, topk=topk)

                bm25_self = self_retrieval_test(
                    chunks=bm25_chunks,
                    search_fn=lambda q, k: bm25_search(q, k),
                    topk=args.topk,
                    sample_n=args.sample_n,
                    seed=args.seed,
                )

                for miss in bm25_self["misses"][:20]:
                    append_jsonl(samples_path, {"bucket": "bm25_self_miss", "item": miss})

                bm25_summary = {
                    "status": "ok",
                    "index_size": len(bm25_chunks),
                    "missing_from_index": len(chunk_id_set - bm25_id_set),
                    "extra_in_index": len(bm25_id_set - chunk_id_set),
                    "self_retrieval": {
                        "sample_n": bm25_self["sample_n"],
                        "topk": bm25_self["topk"],
                        "hit_rate": bm25_self["hit_rate"],
                        "mean_rank": bm25_self["mean_rank"],
                    },
                }
        else:
            bm25_summary = {"status": "missing_index", "bm25_pkl": bm25_pkl}

    # Dense index checks
    dense_summary: Dict[str, Any] = {"status": "skipped"}
    if not args.skip_dense:
        faiss_index = os.path.join(faiss_dir, "faiss.index")
        faiss_meta = os.path.join(faiss_dir, "meta.pkl")
        if dense_index is None:
            dense_summary = {"status": "unavailable", "reason": "dense_index import failed"}
        elif Path(faiss_index).exists() and Path(faiss_meta).exists():
            dense_mod: Any = dense_index
            try:
                obj = dense_mod.load_index(faiss_dir)
            except Exception as exc:
                dense_summary = {"status": "error", "error": str(exc), "faiss_index": faiss_index}
                obj = None
            if obj is None:
                pass
            else:
                dense_chunks = obj.get("chunks", [])
                dense_ids = [c.get("chunk_id", "") for c in dense_chunks if c.get("chunk_id")]
                dense_id_set = set(dense_ids)
                model_name = obj.get("model_name", "")
                def dense_search(query: str, topk: int) -> List[Tuple[float, Dict[str, Any]]]:
                    return dense_mod.search(obj, query, topk=topk)

                dense_self = self_retrieval_test(
                    chunks=dense_chunks,
                    search_fn=lambda q, k: dense_search(q, k),
                    topk=args.topk,
                    sample_n=args.sample_n,
                    seed=args.seed,
                )

                for miss in dense_self["misses"][:20]:
                    append_jsonl(samples_path, {"bucket": "dense_self_miss", "item": miss})

                dense_summary = {
                    "status": "ok",
                    "index_size": len(dense_chunks),
                    "missing_from_index": len(chunk_id_set - dense_id_set),
                    "extra_in_index": len(dense_id_set - chunk_id_set),
                    "model_name": model_name,
                    "model_matches_config": model_name == cfg.get("retrieval", {}).get("embedding_model", ""),
                    "self_retrieval": {
                        "sample_n": dense_self["sample_n"],
                        "topk": dense_self["topk"],
                        "hit_rate": dense_self["hit_rate"],
                        "mean_rank": dense_self["mean_rank"],
                    },
                }
        else:
            dense_summary = {"status": "missing_index", "faiss_index": faiss_index, "faiss_meta": faiss_meta}

    summary = {
        "timestamp": now_tag(),
        "inputs": {
            "config": args.config,
            "chunks_path": chunks_path,
            "docstore_path": docstore_path,
            "bm25_dir": bm25_dir,
            "faiss_dir": faiss_dir,
            "raw_specs_dir": raw_specs_dir,
        },
        "corpus": {
            "pdf_count": len(pdfs),
            "pdf_total_mb": round(sum(pdf_sizes) / (1024 * 1024), 2) if pdf_sizes else 0.0,
            "pdf_max_mb": round(max(pdf_sizes) / (1024 * 1024), 2) if pdf_sizes else 0.0,
        },
        "chunks": {
            "total": len(chunks),
            "length_chars": len_stats.__dict__,
            "length_words": word_stats.__dict__,
            "min_expected_chars": min_expected,
            "max_expected_chars": max_expected,
            "empty": len(empty),
            "too_short": len(too_short),
            "too_long": len(too_long),
            "unknown_clause": len(unknown_clause),
            "dup_chunk_id": len(dup_chunk_ids),
            "dup_text_groups": len(dup_text_groups),
        },
        "docstore": {
            "exists": docstore_exists,
            "rows": docstore_rows,
            "missing_from_docstore": len(missing_in_docstore),
            "extra_in_docstore": len(extra_in_docstore),
        },
        "bm25": bm25_summary,
        "dense": dense_summary,
        "outputs": {
            "summary_json": os.path.join(out_dir, "corpus_index_report.summary.json"),
            "samples_jsonl": samples_path,
        },
    }

    write_json(summary["outputs"]["summary_json"], summary)
    print("[DONE] summary:", summary["outputs"]["summary_json"])
    print("[DONE] samples:", samples_path)


if __name__ == "__main__":
    main()
