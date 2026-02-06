from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

from sentence_transformers import SentenceTransformer

from ..utils import load_yaml, read_jsonl


def get_search_text(c: Dict[str, Any]) -> str:
    title = str(c.get("title", "") or "")
    clause = str(c.get("clause_id", "") or "")
    text = str(c.get("text", "") or "")
    return f"{title}\n{clause}\n{text}".strip()


def build_dense(chunks: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    if faiss is None:
        raise RuntimeError("faiss is not available. Install faiss-cpu or faiss-gpu.")

    model = SentenceTransformer(model_name)

    texts = [get_search_text(c) for c in chunks]
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine/IP ready
    ).astype("float32")

    dim = int(emb.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    return {"index": index, "chunks": chunks, "model_name": model_name}


def save_index(obj: Dict[str, Any], out_dir: str) -> None:
    if faiss is None:
        raise RuntimeError("faiss is not available.")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(obj["index"], os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"chunks": obj["chunks"], "model_name": obj["model_name"]}, f)


def load_index(out_dir: str) -> Dict[str, Any]:
    if faiss is None:
        raise RuntimeError("faiss is not available.")
    idx = faiss.read_index(os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    model = SentenceTransformer(meta["model_name"])
    return {"index": idx, "chunks": meta["chunks"], "model": model, "model_name": meta["model_name"]}


def _passes_filters(c: Dict[str, Any], filters: Dict[str, str]) -> bool:
    for k, v in filters.items():
        if not v:
            continue
        if str(c.get(k, "")).lower() != str(v).lower():
            return False
    return True


def search(
    obj: Dict[str, Any],
    query: str,
    topk: int = 50,
    filters: Optional[Dict[str, str]] = None,
) -> List[Tuple[float, Dict[str, Any]]]:
    if faiss is None:
        raise RuntimeError("faiss is not available.")
    if topk <= 0:
        return []

    model: SentenceTransformer = obj["model"]
    idx = obj["index"]
    chunks: List[Dict[str, Any]] = obj["chunks"]

    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    n = len(chunks)

    # adaptive oversample when filters exist
    mult = 5 if filters else 1
    max_mult = 50 if filters else 1

    pairs: List[Tuple[float, Dict[str, Any]]] = []

    while True:
        k_raw = min(n, topk * mult)
        if k_raw <= 0:
            break

        D, I = idx.search(q, k_raw)
        scores = D[0].tolist()
        inds = I[0].tolist()

        pairs = []
        for s, j in zip(scores, inds):
            if j < 0 or j >= n:
                continue
            c = chunks[j]
            if filters and not _passes_filters(c, filters):
                continue
            pairs.append((float(s), c))
            if len(pairs) >= topk:
                break

        if len(pairs) >= topk or mult >= max_mult or k_raw == n:
            break
        mult *= 2

    return pairs[:topk]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--config", type=str, required=True)

    args = ap.parse_args()
    cfg = load_yaml(args.config)

    if args.cmd == "build":
        if not cfg["retrieval"].get("enable_dense", True):
            print("[SKIP] dense retrieval disabled in config")
            return
        chunks = read_jsonl(cfg["paths"]["chunks_path"])
        out_dir = cfg["paths"]["faiss_dir"]
        model_name = cfg["retrieval"]["embedding_model"]
        obj = build_dense(chunks, model_name=model_name)
        save_index(obj, out_dir)
        print(f"[DONE] Dense FAISS index saved to {out_dir}")


if __name__ == "__main__":
    main()
