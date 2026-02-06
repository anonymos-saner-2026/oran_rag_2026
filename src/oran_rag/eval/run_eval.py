from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import csv
import glob
from typing import Any, Dict, List, Optional, Tuple

from ..rag.answerer import RAGEngine
from ..rag.packer import fetch_neighbors, trim_contexts_by_chars, to_prompt_context
from ..rag.gate import need_fallback
from ..rag.citations import safe_json_loads, clamp_citations
from .metrics import citation_count


# ----------------------------
# OranBench loader
# ----------------------------

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_oranbench(path: str) -> List[Dict[str, Any]]:
    """
    Supports:
      1) JSON list: [ [q, [opt...], "3"], ... ]
      2) JSONL: each line is [q, [opt...], "3"]

    Returns:
      { "question": str, "options": List[str], "answer": str }
    """
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
# Utilities: logging & IO
# ----------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def now_ts() -> str:
    # compact timestamp
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if s is None:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"...(truncated, len={len(s)})"


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    # open+close per write to reduce buffering surprises on NFS
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class TeeLogger:
    """
    Write to both stdout and a txt file, flushing line-by-line.
    """
    def __init__(self, txt_path: str):
        self.txt_path = txt_path
        ensure_dir(os.path.dirname(txt_path) or ".")
        self.f = open(txt_path, "a", encoding="utf-8", buffering=1)  # line buffered

    def close(self) -> None:
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass

    def log(self, msg: str) -> None:
        line = f"[{now_ts()}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


# ----------------------------
# Core: run one QA with full visibility (prompt + raw)
# ----------------------------

def rewrite_query_with_raw(engine: RAGEngine, question_for_rewrite: str) -> Tuple[str, str]:
    """
    Returns (rewritten_query, raw_completion).
    """
    p = engine.prompts.render("query_rewrite.jinja", question=question_for_rewrite)
    raw = engine.llm.generate(
        p,
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.9,
    )
    rewritten = (raw.strip().splitlines()[0] if raw.strip() else question_for_rewrite)
    return rewritten, raw


def answer_with_full_logging(
    engine: RAGEngine,
    question_for_retrieval: str,
    question_for_llm: str,
    filters: Optional[Dict[str, str]],
    top_k: int,
    log: TeeLogger,
    log_prompt: bool,
    log_raw: bool,
    max_prompt_chars: int,
    max_raw_chars: int,
) -> Dict[str, Any]:
    """
    Mirrors engine.answer() but keeps:
      - rewrite raw (optional)
      - final prompt (optional)
      - final raw completion (optional)
    """
    gate_cfg = engine.cfg["gate"]
    pack_cfg = engine.cfg["packing"]
    docstore = engine.cfg["paths"]["docstore_path"]

    rounds = 0
    cur_q = question_for_retrieval
    last_debug: Dict[str, Any] = {}
    rewrite_debug: Dict[str, Any] = {}

    while True:
        rounds += 1
        fused = engine.retrieve(cur_q, filters=filters)
        reranked = engine.rerank(cur_q, fused)

        last_debug = {
            "round": rounds,
            "query": cur_q,
            "fused_top": [(float(s), c.get("chunk_id")) for s, c in fused[:5]],
            "reranked_top": [(float(s), c.get("chunk_id")) for s, c in reranked[:5]],
        }

        if not need_fallback(reranked, float(gate_cfg["min_rerank_score"])):
            break
        if rounds >= int(gate_cfg["max_rounds"]):
            break

        if gate_cfg.get("enable_query_rewrite", True):
            new_q, raw_rewrite = rewrite_query_with_raw(engine, question_for_retrieval)
            rewrite_debug = {
                "enabled": True,
                "raw": truncate(raw_rewrite, max_raw_chars),
                "rewritten": new_q,
            }
            cur_q = new_q
        else:
            break

    # Pack contexts
    seed_chunks = [c for _, c in reranked[:max(3, int(top_k))]]
    contexts = fetch_neighbors(
        db_path=docstore,
        seed_chunks=seed_chunks,
        neighbor_window=int(pack_cfg["neighbor_window"]),
    )
    contexts = trim_contexts_by_chars(contexts, max_chars=int(pack_cfg["max_context_chars"]))
    prompt_contexts = to_prompt_context(contexts)

    prompt = engine.prompts.render(
        "answer_with_citations.jinja",
        question=question_for_llm,
        contexts=prompt_contexts,
    )

    if log_prompt:
        log.log("----- FINAL PROMPT (truncated) -----")
        log.log(truncate(prompt, max_prompt_chars))
        log.log("----- END PROMPT -----")

    raw = engine.llm.generate(
        prompt,
        max_new_tokens=int(engine.cfg["model"]["max_new_tokens"]),
        temperature=float(engine.cfg["model"]["temperature"]),
        top_p=float(engine.cfg["model"]["top_p"]),
    )

    if log_raw:
        log.log("----- RAW LLM OUTPUT (truncated) -----")
        log.log(truncate(raw, max_raw_chars))
        log.log("----- END RAW -----")

    obj = safe_json_loads(raw)
    obj = clamp_citations(obj, max_cites=3)
    obj["_debug"] = last_debug
    if rewrite_debug:
        obj["_rewrite"] = rewrite_debug
    # keep raw optionally for structured output too (truncated)
    if log_raw:
        obj["_raw"] = truncate(raw, max_raw_chars)
    return obj


# ----------------------------
# Worker: evaluate one split with sharding
# ----------------------------

def eval_split_worker(
    engine: RAGEngine,
    split_path: str,
    top_k: int,
    shard_id: int,
    num_shards: int,
    out_jsonl: str,
    log: TeeLogger,
    log_prompt: bool,
    log_raw: bool,
    max_prompt_chars: int,
    max_raw_chars: int,
) -> Dict[str, Any]:
    data = load_oranbench(split_path)
    n_total = len(data)
    if n_total == 0:
        return {"path": split_path, "n": 0, "acc": 0.0, "avg_citations": 0.0}

    indices = list(range(shard_id, n_total, num_shards))

    accs: List[float] = []
    cites: List[int] = []

    log.log(f"[START SPLIT] {os.path.basename(split_path)} total={n_total} my_shard_n={len(indices)} shard={shard_id}/{num_shards}")
    # heartbeat row so you can see jsonl non-empty immediately
    append_jsonl(out_jsonl, {"event": "split_started", "split": split_path, "shard_id": shard_id, "num_shards": num_shards, "time": now_ts()})

    for local_i, i in enumerate(indices, start=1):
        ex = data[i]
        q_text = ex["question"]
        options = ex["options"]
        gold = str(ex["answer"]).strip()

        mcq_question = format_mcq_question(q_text, options)

        log.log("=" * 80)
        log.log(f"[EX] split={os.path.basename(split_path)} global_i={i} local_i={local_i}/{len(indices)}")
        log.log(f"[Q] {q_text}")
        log.log(f"[OPTIONS]\n" + "\n".join(options))
        log.log(f"[GOLD] {gold}")

        out = answer_with_full_logging(
            engine=engine,
            question_for_retrieval=q_text,      # retrieval uses question only
            question_for_llm=mcq_question,      # LLM sees MCQ instruction + options
            filters=None,
            top_k=top_k,
            log=log,
            log_prompt=log_prompt,
            log_raw=log_raw,
            max_prompt_chars=max_prompt_chars,
            max_raw_chars=max_raw_chars,
        )

        pred_text = (out.get("answer", "") or "").strip()
        pred_num = parse_option_number(pred_text)

        em = exact_match_option(pred_num, gold)
        c = citation_count(out)

        accs.append(em)
        cites.append(c)

        # log key outputs
        log.log(f"[PRED_NUM] {pred_num or '?'}")
        log.log(f"[PRED_TEXT] {pred_text!r}")
        log.log(f"[CITATIONS] {out.get('citations', [])}")
        log.log(f"[ACC] {em:.0f}  [CITES] {c}")

        # retrieval debug
        dbg = out.get("_debug", {})
        log.log(f"[DEBUG] round={dbg.get('round')} query={dbg.get('query')}")
        log.log(f"[DEBUG] fused_top={dbg.get('fused_top')}")
        log.log(f"[DEBUG] reranked_top={dbg.get('reranked_top')}")
        if out.get("_rewrite"):
            log.log(f"[REWRITE] {out['_rewrite']}")

        append_jsonl(
            out_jsonl,
            {
                "split": os.path.basename(split_path),
                "global_i": i,
                "question": q_text,
                "options": options,
                "gold": gold,
                "pred_num": pred_num,
                "pred_text": pred_text,
                "acc": em,
                "citations_count": c,
                "citations": out.get("citations", []),
                "_debug": out.get("_debug", {}),
                "_rewrite": out.get("_rewrite", {}),
                "_raw": out.get("_raw", "") if log_raw else "",
            },
        )

    acc = sum(accs) / max(1, len(accs))
    avg_cite = sum(cites) / max(1, len(cites))
    log.log(f"[END SPLIT] {os.path.basename(split_path)} shard_n={len(indices)} acc={acc:.3f} avg_citations={avg_cite:.2f}")
    return {"path": split_path, "n": len(indices), "acc": acc, "avg_citations": avg_cite}


# ----------------------------
# Worker entrypoint (one GPU)
# ----------------------------

def run_worker(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)

    # reset jsonl at start
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        f.write("")

    log = TeeLogger(args.out_log_txt)
    try:
        log.log(f"[WORKER START] shard={args.shard_id}/{args.num_shards} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}")
        log.log(f"[CONFIG] {args.config}")
        log.log(f"[EVAL] eval={args.eval} bench_dir={args.bench_dir} splits={args.splits} top_k={args.top_k}")
        log.log(f"[LOGGING] log_prompt={args.log_prompt} log_raw={args.log_raw} max_prompt_chars={args.max_prompt_chars} max_raw_chars={args.max_raw_chars}")

        engine = RAGEngine.from_config(args.config)

        results: List[Dict[str, Any]] = []
        if args.eval:
            results.append(
                eval_split_worker(
                    engine=engine,
                    split_path=args.eval,
                    top_k=args.top_k,
                    shard_id=args.shard_id,
                    num_shards=args.num_shards,
                    out_jsonl=args.out_jsonl,
                    log=log,
                    log_prompt=args.log_prompt,
                    log_raw=args.log_raw,
                    max_prompt_chars=args.max_prompt_chars,
                    max_raw_chars=args.max_raw_chars,
                )
            )
        else:
            name_map = {"E": "fin_E.json", "M": "fin_M.json", "H": "fin_H.json"}
            split_codes = [s.strip().upper() for s in args.splits.split(",") if s.strip()]
            for s in split_codes:
                path = os.path.join(args.bench_dir, name_map[s])
                results.append(
                    eval_split_worker(
                        engine=engine,
                        split_path=path,
                        top_k=args.top_k,
                        shard_id=args.shard_id,
                        num_shards=args.num_shards,
                        out_jsonl=args.out_jsonl,
                        log=log,
                        log_prompt=args.log_prompt,
                        log_raw=args.log_raw,
                        max_prompt_chars=args.max_prompt_chars,
                        max_raw_chars=args.max_raw_chars,
                    )
                )

        total_n = sum(r["n"] for r in results)
        micro_acc = (sum(r["acc"] * r["n"] for r in results) / max(1, total_n)) if total_n else 0.0
        micro_cite = (sum(r["avg_citations"] * r["n"] for r in results) / max(1, total_n)) if total_n else 0.0

        write_json(
            args.out_summary,
            {
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "results": results,
                "total_n": total_n,
                "micro_acc": micro_acc,
                "micro_avg_citations": micro_cite,
                "out_jsonl": args.out_jsonl,
                "out_log_txt": args.out_log_txt,
            },
        )

        log.log(f"[WORKER DONE] shard={args.shard_id}/{args.num_shards} total_n={total_n} micro_acc={micro_acc:.3f} micro_avg_citations={micro_cite:.2f}")
    finally:
        log.close()


# ----------------------------
# Multi-GPU launcher
# ----------------------------

def run_launcher(args: argparse.Namespace) -> None:
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("[ERROR] --gpus is empty")

    ensure_dir(args.out_dir)

    # Determine split paths
    split_paths: List[str] = []
    if args.eval:
        split_paths = [args.eval]
    else:
        if not args.bench_dir:
            raise SystemExit("[ERROR] Provide --eval <file> or --bench_dir <dir>")
        name_map = {"E": "fin_E.json", "M": "fin_M.json", "H": "fin_H.json"}
        split_codes = [s.strip().upper() for s in args.splits.split(",") if s.strip()]
        for s in split_codes:
            if s not in name_map:
                raise SystemExit(f"[ERROR] Unknown split '{s}'. Use E,M,D.")
            p = os.path.join(args.bench_dir, name_map[s])
            if not os.path.exists(p):
                raise SystemExit(f"[ERROR] Split file not found: {p}")
            split_paths.append(p)

    procs: List[subprocess.Popen] = []
    summaries: List[str] = []

    for shard_id, gpu in enumerate(gpus):
        prefix = f"shard_{shard_id}_gpu{gpu}"
        out_jsonl = os.path.join(args.out_dir, f"{prefix}.jsonl")
        out_summary = os.path.join(args.out_dir, f"{prefix}.summary.json")
        out_log_txt = os.path.join(args.out_dir, f"{prefix}.log.txt")
        out_subproc_log = os.path.join(args.out_dir, f"{prefix}.subprocess.log.txt")
        summaries.append(out_summary)

        cmd = [
            sys.executable, "-m", "oran_rag.eval.run_eval",
            "--config", args.config,
            "--top_k", str(args.top_k),
            "--out_dir", args.out_dir,
            "--worker",
            "--shard_id", str(shard_id),
            "--num_shards", str(len(gpus)),
            "--out_jsonl", out_jsonl,
            "--out_summary", out_summary,
            "--out_log_txt", out_log_txt,
            "--log_prompt", "1" if args.log_prompt else "0",
            "--log_raw", "1" if args.log_raw else "0",
            "--max_prompt_chars", str(args.max_prompt_chars),
            "--max_raw_chars", str(args.max_raw_chars),
        ]
        if args.eval:
            cmd += ["--eval", args.eval]
        else:
            cmd += ["--bench_dir", args.bench_dir, "--splits", args.splits]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"

        with open(out_subproc_log, "w", encoding="utf-8") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
            procs.append(p)

        print(f"[LAUNCH] gpu={gpu} shard_id={shard_id} log_txt={out_log_txt} subproc_log={out_subproc_log}", flush=True)

    failed = False
    for p in procs:
        rc = p.wait()
        if rc != 0:
            failed = True

    if failed:
        print("[ERROR] One or more workers failed. Check *.subprocess.log.txt and *.log.txt in:", args.out_dir, flush=True)
        raise SystemExit(2)

    # Aggregate summaries
    all_rows: List[Dict[str, Any]] = []
    for sp in summaries:
        with open(sp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        for row in obj.get("results", []):
            all_rows.append(row)

    by_split: Dict[str, Dict[str, float]] = {}
    for r in all_rows:
        key = os.path.basename(r["path"])
        by_split.setdefault(key, {"n": 0.0, "acc_sum": 0.0, "cite_sum": 0.0})
        by_split[key]["n"] += float(r["n"])
        by_split[key]["acc_sum"] += float(r["acc"]) * float(r["n"])
        by_split[key]["cite_sum"] += float(r["avg_citations"]) * float(r["n"])

    total_n = 0
    sum_acc = 0.0
    sum_cite = 0.0

    print("\n====================", flush=True)
    for key in sorted(by_split.keys()):
        n = int(by_split[key]["n"])
        acc = by_split[key]["acc_sum"] / max(1, n)
        cite = by_split[key]["cite_sum"] / max(1, n)
        print(f"[SPLIT] {key}: n={n} acc={acc:.3f} avg_citations={cite:.2f}", flush=True)
        total_n += n
        sum_acc += acc * n
        sum_cite += cite * n

    micro_acc = sum_acc / max(1, total_n)
    micro_cite = sum_cite / max(1, total_n)
    print(f"[OVERALL] n={total_n} micro_acc={micro_acc:.3f} micro_avg_citations={micro_cite:.2f}", flush=True)

    write_json(
        os.path.join(args.out_dir, "aggregate.summary.json"),
        {
            "total_n": total_n,
            "micro_acc": micro_acc,
            "micro_avg_citations": micro_cite,
            "by_split": {
                k: {
                    "n": int(v["n"]),
                    "acc": (v["acc_sum"] / max(1, int(v["n"]))),
                    "avg_citations": (v["cite_sum"] / max(1, int(v["n"]))),
                }
                for k, v in by_split.items()
            },
            "out_dir": args.out_dir,
        },
    )

    # Export per-split CSVs by aggregating all per-shard JSONL rows
    try:
        all_rows: List[Dict[str, Any]] = []
        jsonl_pattern = os.path.join(args.out_dir, "*.jsonl")
        for jl in glob.glob(jsonl_pattern):
            try:
                with open(jl, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        # skip heartbeat rows
                        if obj.get("event") == "split_started":
                            continue
                        # only rows that look like evaluation examples
                        if "split" in obj and "gold" in obj and "pred_text" in obj:
                            all_rows.append(obj)
            except Exception:
                # ignore per-shard read errors
                continue

        # group by split basename
        by_split_rows: Dict[str, List[Dict[str, Any]]] = {}
        for r in all_rows:
            key = os.path.basename(r.get("split", "unknown"))
            by_split_rows.setdefault(key, []).append(r)

        for split_name, rows in by_split_rows.items():
            csv_path = os.path.join(args.out_dir, f"{split_name}.csv")
            try:
                with open(csv_path, "w", encoding="utf-8", newline="") as cf:
                    writer = csv.writer(cf)
                    # header
                    writer.writerow(["global_i", "question", "options", "gold", "pred_num", "pred_text", "acc", "citations_count", "citations"])
                    for r in rows:
                        opts = r.get("options", [])
                        opts_s = json.dumps(opts, ensure_ascii=False)
                        citations = json.dumps(r.get("citations", []), ensure_ascii=False)
                        writer.writerow([
                            r.get("global_i", ""),
                            r.get("question", ""),
                            opts_s,
                            r.get("gold", ""),
                            r.get("pred_num", ""),
                            r.get("pred_text", ""),
                            r.get("acc", ""),
                            r.get("citations_count", ""),
                            citations,
                        ])
                print(f"[OK] Wrote CSV -> {csv_path}", flush=True)
            except Exception:
                print(f"[WARN] Failed to write CSV for split {split_name}", flush=True)
    except Exception:
        print("[WARN] CSV export failed (continuing)", flush=True)

    # Generate markdown report with experiment metadata
    try:
        md_path = os.path.join(args.out_dir, "EXPERIMENT_REPORT.md")
        
        # Read config file for settings
        config_content = ""
        try:
            with open(args.config, "r", encoding="utf-8") as cf:
                config_content = cf.read()
        except Exception:
            config_content = f"[Config file not readable: {args.config}]"
        
        # Read aggregate summary
        summary_data = {}
        summary_path = os.path.join(args.out_dir, "aggregate.summary.json")
        try:
            with open(summary_path, "r", encoding="utf-8") as sf:
                summary_data = json.load(sf)
        except Exception:
            summary_data = {"error": "Summary not available"}
        
        # Build markdown
        md_lines = [
            "# O-RAN RAG Evaluation Report",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Experiment Settings",
            "",
            f"- **Config File:** `{args.config}`",
            f"- **Output Directory:** `{args.out_dir}`",
            f"- **GPUs Used:** {args.gpus}",
            f"- **Splits:** {args.splits}",
            f"- **Top-K Retrieval:** {args.top_k}",
            f"- **Benchmark Dir:** {args.bench_dir if args.bench_dir else args.eval}",
            "",
        ]
        
        # Add key sections from config
        md_lines.extend([
            "## Configuration Details",
            "",
            "```yaml",
        ])
        md_lines.extend(config_content.split("\n"))
        md_lines.extend([
            "```",
            "",
        ])
        
        # Add metrics
        md_lines.extend([
            "## Results Summary",
            "",
        ])
        
        if summary_data.get("total_n"):
            total_n = summary_data.get("total_n", 0)
            micro_acc = summary_data.get("micro_acc", 0)
            micro_cite = summary_data.get("micro_avg_citations", 0)
            
            md_lines.extend([
                f"- **Total Questions:** {total_n}",
                f"- **Overall Accuracy:** {micro_acc * 100:.2f}%",
                f"- **Avg Citations per Answer:** {micro_cite:.2f}",
                "",
                "### Per-Split Performance",
                "",
            ])
            
            by_split = summary_data.get("by_split", {})
            for split_name in sorted(by_split.keys()):
                split_data = by_split[split_name]
                n = split_data.get("n", 0)
                acc = split_data.get("acc", 0)
                citations = split_data.get("avg_citations", 0)
                md_lines.append(f"- **{split_name}:** {n} questions | {acc * 100:.2f}% accuracy | {citations:.2f} avg citations")
        else:
            md_lines.append("*(Results not yet available)*")
            md_lines.append("")
        
        # Add output files
        csv_files = sorted(glob.glob(os.path.join(args.out_dir, "*.csv")))
        if csv_files:
            md_lines.extend([
                "## Output Files",
                "",
                "### CSV Results (Per-Split Model Predictions)",
                "",
            ])
            for csv_f in csv_files:
                csv_name = os.path.basename(csv_f)
                md_lines.append(f"- `{csv_name}` - Model predictions vs. golden answers")
            md_lines.append("")
            md_lines.extend([
                "### Other Output Files",
                "",
                f"- `aggregate.summary.json` - Aggregated metrics across all GPUs",
                f"- `all_workers.log.txt` - Combined logs from all workers",
                f"- `shard_*_gpu*.jsonl` - Per-GPU evaluation details (JSONL format)",
                f"- `shard_*_gpu*.log.txt` - Per-GPU log files",
                "",
            ])
        
        # Write markdown
        with open(md_path, "w", encoding="utf-8") as mf:
            mf.write("\n".join(md_lines))
        
        print(f"[OK] Wrote markdown report -> {md_path}", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to generate markdown report: {e}", flush=True)

    # Optional: create a combined txt by concatenating per-worker log.txt
    combined = os.path.join(args.out_dir, "all_workers.log.txt")
    with open(combined, "w", encoding="utf-8") as out_f:
        for shard_id, gpu in enumerate([g.strip() for g in args.gpus.split(",") if g.strip()]):
            p = os.path.join(args.out_dir, f"shard_{shard_id}_gpu{gpu}.log.txt")
            if os.path.exists(p):
                out_f.write(f"\n\n===== {os.path.basename(p)} =====\n")
                out_f.write(_read_text(p))
    print(f"[OK] Combined worker logs -> {combined}", flush=True)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)

    ap.add_argument("--eval", type=str, default=None, help="Single split file (fin_E.json)")
    ap.add_argument("--bench_dir", type=str, default=None, help="Dir containing fin_E.json/fin_M.json/fin_H.json")
    ap.add_argument("--splits", type=str, default="E,M,D", help="Comma-separated: E,M,D")
    ap.add_argument("--top_k", type=int, default=10)

    # Multi-GPU launcher
    ap.add_argument("--gpus", type=str, default="", help='Comma-separated GPU ids, e.g. "0,1,2,3"')
    ap.add_argument("--out_dir", type=str, default="data/reports/eval_mgpu", help="Output directory")

    # Logging controls (txt log in each worker)
    ap.add_argument("--log_prompt", type=str, default="0", help="1 to log final prompt into txt (truncated)")
    ap.add_argument("--log_raw", type=str, default="1", help="1 to log raw LLM output into txt (truncated)")
    ap.add_argument("--max_prompt_chars", type=int, default=4000)
    ap.add_argument("--max_raw_chars", type=int, default=2000)

    # Internal worker flags
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shard_id", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--num_shards", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--out_jsonl", type=str, default="", help=argparse.SUPPRESS)
    ap.add_argument("--out_summary", type=str, default="", help=argparse.SUPPRESS)
    ap.add_argument("--out_log_txt", type=str, default="", help=argparse.SUPPRESS)

    args = ap.parse_args()
    args.log_prompt = str(args.log_prompt).strip() in ("1", "true", "True", "yes", "Y")
    args.log_raw = str(args.log_raw).strip() in ("1", "true", "True", "yes", "Y")

    if args.worker:
        if not args.out_jsonl or not args.out_summary or not args.out_log_txt:
            raise SystemExit("[ERROR] worker mode requires --out_jsonl --out_summary --out_log_txt")
        run_worker(args)
        return

    # Launcher mode if --gpus provided
    if args.gpus.strip():
        run_launcher(args)
        return

    # Single-process fallback
    ensure_dir(args.out_dir)
    prefix = "single"
    out_jsonl = os.path.join(args.out_dir, f"{prefix}.jsonl")
    out_summary = os.path.join(args.out_dir, f"{prefix}.summary.json")
    out_log_txt = os.path.join(args.out_dir, f"{prefix}.log.txt")

    # make it behave like a worker
    args.worker = True
    args.shard_id = 0
    args.num_shards = 1
    args.out_jsonl = out_jsonl
    args.out_summary = out_summary
    args.out_log_txt = out_log_txt
    run_worker(args)


if __name__ == "__main__":
    main()
