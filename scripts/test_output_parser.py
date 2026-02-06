from __future__ import annotations

import argparse
import json
import random
import re
from typing import Any, Dict, List, Tuple

# Import đúng path theo project bạn
from oran_rag.rag.citations import safe_json_loads, clamp_citations


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_valid_schema(obj: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not_dict"
    if set(obj.keys()) != {"answer", "citations"}:
        # allow extra keys? -> NO, we want strict
        # but safe_json_loads returns exactly answer/citations in our patch
        return False, f"bad_keys={sorted(obj.keys())}"
    if not isinstance(obj.get("answer", ""), str):
        return False, "answer_not_str"
    if not isinstance(obj.get("citations", []), list):
        return False, "citations_not_list"
    for c in obj["citations"]:
        if not isinstance(c, dict):
            return False, "cite_not_dict"
        if set(c.keys()) != {"chunk_id", "quote"}:
            return False, "cite_bad_keys"
        if not isinstance(c["chunk_id"], str) or not isinstance(c["quote"], str):
            return False, "cite_val_not_str"
    return True, "ok"


def classify_parse(raw: str, parsed: Dict[str, Any]) -> str:
    """
    Heuristic classification:
      - "clean": raw is a pure JSON object already
      - "extracted_or_fixed": parser had to extract/repair JSON
      - "salvaged": parser fell back to regex salvage (common when no JSON object)
    """
    raw_s = (raw or "").strip()

    # clean JSON: must start with '{' and end with '}' and json.loads works
    if raw_s.startswith("{") and raw_s.endswith("}"):
        try:
            json.loads(raw_s)
            return "clean"
        except Exception:
            pass

    # If the raw contains a JSON object somewhere, it's likely extracted/fixed
    if "{" in raw_s and "}" in raw_s:
        return "extracted_or_fixed"

    # Otherwise it's salvage by regex
    return "salvaged"


def gen_synthetic_cases() -> List[str]:
    """
    Simulate common LLM bad outputs.
    """
    cases = []

    # Perfect
    cases.append('{"answer":"2","citations":[{"chunk_id":"DOC::1","quote":"O-CU: O-RAN Central Unit"}]}')

    # Code fence
    cases.append('```json\n{"answer":"1","citations":[]}\n```')

    # Extra text before JSON
    cases.append('Sure! Here is the result:\n{"answer":"3","citations":[]}')

    # Extra text after JSON
    cases.append('{"answer":"4","citations":[]}\nDone.')

    # Single quotes + bare keys
    cases.append("{answer: '2', citations: [{'chunk_id': 'X', 'quote': 'abc'}]}")

    # Trailing commas
    cases.append('{"answer":"2","citations":[{"chunk_id":"X","quote":"abc",},],}')

    # Not JSON, but contains Answer: 2
    cases.append("Answer: 2\nCitations: none")

    # Totally messy but has a number
    cases.append("I think it is option 3 because ...")

    # Missing keys
    cases.append('{"ans":"2"}')

    # Empty
    cases.append("")

    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, default="", help="Input jsonl that contains a field '_raw' (from run_eval)")
    ap.add_argument("--raw_field", type=str, default="_raw", help="Which field contains raw LLM output")
    ap.add_argument("--n", type=int, default=0, help="Limit number of rows to test (0=all)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--show_bad", type=int, default=8, help="Show up to K failing cases")
    ap.add_argument("--synthetic", action="store_true", help="Run synthetic cases instead of reading jsonl")
    args = ap.parse_args()

    random.seed(args.seed)

    raws: List[str] = []

    if args.synthetic:
        raws = gen_synthetic_cases()
        print(f"[INFO] Using synthetic cases: {len(raws)}")
    else:
        if not args.in_jsonl:
            raise SystemExit("[ERROR] Provide --in_jsonl or use --synthetic")
        rows = read_jsonl(args.in_jsonl)
        if args.n and args.n > 0:
            rows = rows[: args.n]
        for r in rows:
            raw = r.get(args.raw_field, "")
            if raw is None:
                raw = ""
            raws.append(str(raw))
        print(f"[INFO] Loaded {len(raws)} raw outputs from: {args.in_jsonl} (field={args.raw_field})")

    stats = {
        "total": 0,
        "schema_ok": 0,
        "schema_fail": 0,
        "clean": 0,
        "extracted_or_fixed": 0,
        "salvaged": 0,
        "empty_raw": 0,
        "answer_missing": 0,
    }

    bad_examples: List[Tuple[int, str, Dict[str, Any], str]] = []

    for i, raw in enumerate(raws):
        stats["total"] += 1
        if not raw.strip():
            stats["empty_raw"] += 1

        parsed = safe_json_loads(raw)
        parsed = clamp_citations(parsed, max_cites=3)

        ok, reason = is_valid_schema(parsed)
        mode = classify_parse(raw, parsed)

        if mode in stats:
            stats[mode] += 1

        if not parsed.get("answer", "").strip():
            stats["answer_missing"] += 1

        if ok:
            stats["schema_ok"] += 1
        else:
            stats["schema_fail"] += 1
            if len(bad_examples) < args.show_bad:
                bad_examples.append((i, raw, parsed, reason))

    # Print report
    print("\n==================== PARSER TEST REPORT ====================")
    for k in ["total", "schema_ok", "schema_fail", "clean", "extracted_or_fixed", "salvaged", "empty_raw", "answer_missing"]:
        print(f"{k:>18}: {stats[k]}")
    if stats["total"] > 0:
        print(f"{'schema_ok_rate':>18}: {stats['schema_ok']/stats['total']:.4f}")
        print(f"{'answer_missing_rate':>18}: {stats['answer_missing']/stats['total']:.4f}")

    if bad_examples:
        print("\n==================== FAILING EXAMPLES ====================")
        for (idx, raw, parsed, reason) in bad_examples:
            raw_trunc = raw.strip().replace("\n", "\\n")
            if len(raw_trunc) > 240:
                raw_trunc = raw_trunc[:240] + "...(trunc)"
            print(f"\n[CASE #{idx}] reason={reason}")
            print(f"RAW:    {raw_trunc}")
            print(f"PARSED: {json.dumps(parsed, ensure_ascii=False)}")
    else:
        print("\n✅ No schema failures detected.")

    print("===========================================================\n")


if __name__ == "__main__":
    main()
