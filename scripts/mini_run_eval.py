from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from oran_rag.rag.answerer import RAGEngine
from oran_rag.rag.citations import safe_json_loads, clamp_citations
from oran_rag.rag.packer import fetch_neighbors, trim_contexts_by_chars, to_prompt_context


# ----------------------------
# OranBench loader (same behavior)
# ----------------------------

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_oranbench(path: str) -> List[Dict[str, Any]]:
    raw = _read_text(path).strip()
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
            opts = ex[1]
            gold = str(ex[2]).strip()
            if not isinstance(opts, list):
                opts = [str(opts)]
            opts = [str(o).strip() for o in opts]
            if q and gold:
                out.append({"question": q, "options": opts, "answer": gold})
        elif isinstance(ex, dict):
            q = str(ex.get("question", "")).strip()
            opts = ex.get("options", ex.get("choices", []))
            gold = str(ex.get("answer", ex.get("label", ""))).strip()
            if not isinstance(opts, list):
                opts = [str(opts)]
            opts = [str(o).strip() for o in opts]
            if q and gold:
                out.append({"question": q, "options": opts, "answer": gold})
    return out


# ----------------------------
# MCQ formatting + parsing
# ----------------------------

_MCQ_NUM_RE = re.compile(r"\b([1-9][0-9]?)\b")

def format_mcq_question(question: str, options: List[str]) -> str:
    opts_txt = "\n".join(options)
    return (
        "You are answering a multiple-choice question about O-RAN specifications.\n"
        "Return a JSON object with exactly this schema:\n"
        "{\n"
        '  "answer": "<option_number>",\n'
        '  "citations": [{"chunk_id": "...", "quote": "..."}]\n'
        "}\n"
        "Rules:\n"
        "- answer must be only the number as a string, e.g. \"1\".\n"
        "- citations: up to 3 items, MUST be supported by provided context.\n\n"
        f"Question: {question}\n"
        f"Options:\n{opts_txt}\n"
    )

def parse_option_number(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.isdigit():
        return t
    m = _MCQ_NUM_RE.search(t)
    return m.group(1) if m else ""

def exact_match_option(pred_num: str, gold_num: str) -> float:
    return 1.0 if (pred_num or "").strip() == (gold_num or "").strip() else 0.0


# ----------------------------
# IO helpers
# ----------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def now_ts_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ----------------------------
# Mini runner
# ----------------------------

def run_one(engine: RAGEngine, q_text: str, options: List[str], top_k: int, keep_raw: bool) -> Dict[str, Any]:
    """
    Similar to engine.answer(), but:
      - LLM gets MCQ wrapper
      - we also keep retrieval debug and optionally raw output
    """
    # retrieval
    fused = engine.retrieve(q_text, filters=None)
    reranked = engine.rerank(q_text, fused)

    # pack contexts (same as engine.answer)
    pack_cfg = engine.cfg["packing"]
    docstore = engine.cfg["paths"]["docstore_path"]

    seed_chunks = [c for _, c in reranked[:max(3, int(top_k))]]
    contexts = fetch_neighbors(
        db_path=docstore,
        seed_chunks=seed_chunks,
        neighbor_window=int(pack_cfg["neighbor_window"]),
    )
    contexts = trim_contexts_by_chars(contexts, max_chars=int(pack_cfg["max_context_chars"]))
    prompt_contexts = to_prompt_context(contexts)

    mcq_question = format_mcq_question(q_text, options)

    prompt = engine.prompts.render(
        "answer_with_citations.jinja",
        question=mcq_question,
        contexts=prompt_contexts,
    )

    raw = engine.llm.generate(
        prompt,
        max_new_tokens=int(engine.cfg["model"]["max_new_tokens"]),
        temperature=float(engine.cfg["model"]["temperature"]),
        top_p=float(engine.cfg["model"]["top_p"]),
    )

    obj = safe_json_loads(raw)
    obj = clamp_citations(obj, max_cites=3)

    obj["_debug"] = {
        "query": q_text,
        "fused_top": [(float(s), c.get("chunk_id")) for s, c in fused[:5]],
        "reranked_top": [(float(s), c.get("chunk_id")) for s, c in reranked[:5]],
    }

    if keep_raw:
        obj["_raw"] = raw[:2000]  # cap
        obj["_prompt"] = prompt[:3000]  # cap

    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--eval", type=str, required=True, help="Path to fin_E.json (or any split file)")
    ap.add_argument("--n", type=int, default=10, help="Number of samples to run")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--start", type=int, default=-1, help="If >=0, run range [start, start+n)")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out_dir", type=str, default="data/reports/mini_eval")
    ap.add_argument("--keep_raw", action="store_true", help="Store truncated raw + prompt into jsonl for debugging")
    ap.add_argument("--verbose", action="store_true", help="Print per-example details to stdout")
    args = ap.parse_args()

    random.seed(args.seed)

    ensure_dir(args.out_dir)
    ts = now_ts_compact()
    out_jsonl = os.path.join(args.out_dir, f"mini_eval_{ts}.jsonl")
    out_summary = os.path.join(args.out_dir, f"mini_eval_{ts}.summary.json")

    data = load_oranbench(args.eval)
    if not data:
        raise SystemExit(f"[ERROR] No data loaded from {args.eval}")

    n_total = len(data)
    n = min(max(1, args.n), n_total)

    if args.start is not None and args.start >= 0:
        start = min(args.start, n_total - 1)
        indices = list(range(start, min(n_total, start + n)))
    else:
        indices = random.sample(range(n_total), k=n)

    engine = RAGEngine.from_config(args.config)

    accs: List[float] = []
    parse_fail = 0

    print(f"[MINI_EVAL] split={os.path.basename(args.eval)} total={n_total} running={len(indices)} out={out_jsonl}")

    for j, i in enumerate(tqdm(indices, desc="mini_eval", unit="ex"), start=1):
        ex = data[i]
        q_text = ex["question"]
        options = ex["options"]
        gold = str(ex["answer"]).strip()

        out = run_one(engine, q_text=q_text, options=options, top_k=args.top_k, keep_raw=args.keep_raw)

        pred_text = (out.get("answer", "") or "").strip()
        pred_num = parse_option_number(pred_text)
        em = exact_match_option(pred_num, gold)
        accs.append(em)

        # detect parse issues: if answer missing entirely, treat as parse/model fail
        if not pred_num:
            parse_fail += 1

        row = {
            "global_i": i,
            "question": q_text,
            "options": options,
            "gold": gold,
            "pred_num": pred_num,
            "pred_text": pred_text,
            "acc": em,
            "citations": out.get("citations", []),
            "_debug": out.get("_debug", {}),
        }
        if args.keep_raw:
            row["_raw"] = out.get("_raw", "")
            row["_prompt"] = out.get("_prompt", "")

        append_jsonl(out_jsonl, row)

        if args.verbose:
            print("\n" + "=" * 90)
            print(f"[{j}/{len(indices)}] i={i}  ACC={int(em)}  pred={pred_num or '?'}  gold={gold}")
            print(f"Q: {q_text}")
            print("Top reranked chunk_ids:", row["_debug"].get("reranked_top"))
            print("Citations:", row["citations"])

    micro_acc = sum(accs) / max(1, len(accs))
    summary = {
        "split": os.path.basename(args.eval),
        "n_total": n_total,
        "n_run": len(indices),
        "indices": indices,
        "micro_acc": micro_acc,
        "missing_answer_rate": (parse_fail / max(1, len(indices))),
        "out_jsonl": out_jsonl,
    }
    write_json(out_summary, summary)

    print("\n[RESULT]")
    print(f"  micro_acc = {micro_acc:.3f}  (n={len(indices)})")
    print(f"  missing_answer_rate = {summary['missing_answer_rate']:.3f}")
    print(f"  jsonl = {out_jsonl}")
    print(f"  summary = {out_summary}")


if __name__ == "__main__":
    main()
