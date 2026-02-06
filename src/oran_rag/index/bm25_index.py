from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from ..utils import load_yaml, read_jsonl


# -----------------------------
# Text & Tokenization utilities
# -----------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/\.][A-Za-z0-9]+)*")


def get_search_text(c: Dict[str, Any]) -> str:
    """
    Unified retrieval text across BM25/Dense/Rerank:
    include title + clause_id + body text to help MCQ about terms/abbrev.
    """
    title = str(c.get("title", "") or "")
    clause = str(c.get("clause_id", "") or "")
    text = str(c.get("text", "") or "")
    out = f"{title}\n{clause}\n{text}".strip()
    return out


def tokenize_oran(s: str) -> List[str]:
    """
    O-RAN aware tokenizer:
    - keep hyphenated tokens: O-CU-CP, Near-RT, etc.
    - also add split parts and a collapsed variant (OCUCP) to improve matching.
    """
    raw = _TOKEN_RE.findall(s)
    out: List[str] = []
    for t in raw:
        tl = t.lower()
        out.append(tl)

        # add variants
        if any(ch in t for ch in "-_/." ):
            parts = re.split(r"[-_/\.]", t)
            parts = [p for p in parts if p]
            out.extend([p.lower() for p in parts])
            collapsed = re.sub(r"[-_/\.]", "", t).lower()
            if collapsed and collapsed != tl:
                out.append(collapsed)
    return out


# -----------------------------
# Index build/load
# -----------------------------
def build_bm25(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    corpus_tokens = [tokenize_oran(get_search_text(c)) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    return {
        "bm25": bm25,
        "chunks": chunks,
        "corpus_tokens": corpus_tokens,  # keep for debug/rebuild convenience
        "tokenizer": "tokenize_oran_v1",
        "search_text": "title+clause+text",
    }


def save_index(index: Dict[str, Any], out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(index, f)


def load_index(dir_path: str) -> Dict[str, Any]:
    with open(os.path.join(dir_path, "bm25.pkl"), "rb") as f:
        return pickle.load(f)


# -----------------------------
# Search
# -----------------------------
def _passes_filters(c: Dict[str, Any], filters: Dict[str, str]) -> bool:
    for k, v in filters.items():
        if not v:
            continue
        if str(c.get(k, "")).lower() != str(v).lower():
            return False
    return True


def search(
    index: Dict[str, Any],
    query: str,
    topk: int = 50,
    filters: Optional[Dict[str, str]] = None,
) -> List[Tuple[float, Dict[str, Any]]]:
    if topk <= 0:
        return []

    bm25: BM25Okapi = index["bm25"]
    chunks: List[Dict[str, Any]] = index["chunks"]

    qtok = tokenize_oran(query)
    scores = np.asarray(bm25.get_scores(qtok), dtype=np.float32)

    # No filter: fast top-k selection without sorting all.
    if not filters:
        n = len(chunks)
        k = min(topk, n)
        if k <= 0:
            return []
        if k == n:
            idxs = np.argsort(-scores)
        else:
            idxs = np.argpartition(scores, -k)[-k:]
            idxs = idxs[np.argsort(-scores[idxs])]
        return [(float(scores[i]), chunks[i]) for i in idxs]

    # With filters: oversample then filter to avoid empty result after filtering.
    # adaptive multiplier
    n = len(chunks)
    mult = 5
    out: List[Tuple[float, Dict[str, Any]]] = []

    while True:
        k_raw = min(n, topk * mult)
        if k_raw <= 0:
            break

        idxs = np.argpartition(scores, -k_raw)[-k_raw:]
        idxs = idxs[np.argsort(-scores[idxs])]

        out = []
        for i in idxs:
            c = chunks[int(i)]
            if _passes_filters(c, filters):
                out.append((float(scores[int(i)]), c))
                if len(out) >= topk:
                    break

        if len(out) >= topk or mult >= 50 or k_raw == n:
            break
        mult *= 2

    return out[:topk]


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--config", type=str, required=True)

    args = ap.parse_args()
    cfg = load_yaml(args.config)

    if args.cmd == "build":
        chunks_path = cfg["paths"]["chunks_path"]
        out_dir = cfg["paths"]["bm25_dir"]
        chunks = read_jsonl(chunks_path)
        idx = build_bm25(chunks)
        save_index(idx, out_dir)
        print(f"[DONE] BM25 index saved to {out_dir}")


if __name__ == "__main__":
    main()
