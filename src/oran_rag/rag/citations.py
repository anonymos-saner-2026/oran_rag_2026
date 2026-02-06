from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# ----------------------------
# Helpers: robust JSON extraction
# ----------------------------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Try to capture first JSON object {...} even if there is extra text around it.
# This uses a simple brace scan (safer than regex for nested braces).
def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip()

    # If it's fenced, prefer inside fence
    m = _CODE_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    # Find first '{' and scan to matching '}'.
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def _strip_markdown(text: str) -> str:
    if not text:
        return ""
    t = text.strip()

    # remove code fences (keep inner)
    m = _CODE_FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()

    # remove common leading junk like "Sure:" or "Output:" lines
    # keep from first '{' if exists
    j = t.find("{")
    if j >= 0:
        t = t[j:].strip()
    return t


def _fix_common_json_issues(s: str) -> str:
    """
    Best-effort fixes for common LLM JSON errors:
    - single quotes -> double quotes (only when it looks like JSON-ish)
    - trailing commas before } or ]
    - unquoted keys (limited)
    Note: we keep this conservative to avoid mangling legitimate text.
    """
    if not s:
        return s

    t = s.strip()

    # Remove trailing commas:  {...,} or [...,]
    t = re.sub(r",\s*([}\]])", r"\1", t)

    # Convert smart quotes to normal quotes
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")

    # If it looks like using single quotes for keys/strings, convert to double quotes.
    # This is heuristic: if there is no double quote at all but there are many single quotes.
    if ('"' not in t) and (t.count("'") >= 2):
        t = t.replace("'", '"')

    # Quote bare keys: {answer: "1"} -> {"answer": "1"}
    # Only for simple word keys (letters/underscores).
    t = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', t)

    return t


# ----------------------------
# Public API
# ----------------------------

def safe_json_loads(raw: str) -> Dict[str, Any]:
    """
    Robustly parse model output into a dict of schema:
      {
        "answer": "<option_number_as_string>",
        "citations": [{"chunk_id": "...", "quote": "..."}]
      }

    Never raises. If parsing fails, returns a minimal dict with empty fields.
    """
    if raw is None:
        return {"answer": "", "citations": []}

    text = str(raw).strip()
    if not text:
        return {"answer": "", "citations": []}

    # 1) strip markdown/code fences
    t = _strip_markdown(text)

    # 2) extract first JSON object if there is surrounding noise
    candidate = _extract_first_json_object(t) or t

    # 3) try strict json
    obj: Any = None
    try:
        obj = json.loads(candidate)
    except Exception:
        # 4) try fix common issues then parse
        fixed = _fix_common_json_issues(candidate)
        try:
            obj = json.loads(fixed)
        except Exception:
            obj = None

    # 5) If still not parsed, try regex salvage
    if not isinstance(obj, dict):
        return _salvage_with_regex(text)

    # Normalize schema
    ans = obj.get("answer", "")
    if ans is None:
        ans = ""
    ans = str(ans).strip()

    cites = obj.get("citations", [])
    if cites is None:
        cites = []
    if not isinstance(cites, list):
        cites = []

    norm_cites: List[Dict[str, str]] = []
    for c in cites[:20]:
        if isinstance(c, dict):
            chunk_id = str(c.get("chunk_id", "") or "").strip()
            quote = str(c.get("quote", "") or "").strip()
            if chunk_id or quote:
                norm_cites.append({"chunk_id": chunk_id, "quote": quote})
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            chunk_id = str(c[0] or "").strip()
            quote = str(c[1] or "").strip()
            if chunk_id or quote:
                norm_cites.append({"chunk_id": chunk_id, "quote": quote})

    return {"answer": ans, "citations": norm_cites}


def _salvage_with_regex(text: str) -> Dict[str, Any]:
    """
    Last-resort extraction:
    - answer: first standalone number 1-99 (prefer patterns like "answer": "3" or Answer: 3)
    - citations: try to extract chunk_id + quote pairs if present
    """
    t = text.strip()

    # answer: prefer JSON-like keys
    m = re.search(r'"answer"\s*:\s*"(\d{1,2})"', t)
    if not m:
        m = re.search(r'"answer"\s*:\s*(\d{1,2})', t)
    if not m:
        m = re.search(r"\bAnswer\b\s*[:=]\s*(\d{1,2})\b", t, flags=re.IGNORECASE)
    if not m:
        # fallback: first isolated number (1..99)
        m = re.search(r"\b([1-9]\d?)\b", t)

    ans = m.group(1) if m else ""

    # citations: attempt extract objects like {"chunk_id": "...", "quote": "..."}
    cite_objs: List[Dict[str, str]] = []
    for cm in re.finditer(r'"chunk_id"\s*:\s*"([^"]*)"\s*,\s*"quote"\s*:\s*"([^"]*)"', t):
        cite_objs.append({"chunk_id": cm.group(1).strip(), "quote": cm.group(2).strip()})
        if len(cite_objs) >= 3:
            break

    return {"answer": ans, "citations": cite_objs}


def clamp_citations(obj: Dict[str, Any], max_cites: int = 3, max_quote_chars: int = 260) -> Dict[str, Any]:
    """
    Clamp citations to safe bounds (count and quote length).
    """
    if not isinstance(obj, dict):
        return {"answer": "", "citations": []}

    ans = obj.get("answer", "")
    if ans is None:
        ans = ""
    ans = str(ans).strip()

    cites = obj.get("citations", [])
    if not isinstance(cites, list):
        cites = []

    out_cites: List[Dict[str, str]] = []
    for c in cites:
        if not isinstance(c, dict):
            continue
        chunk_id = str(c.get("chunk_id", "") or "").strip()
        quote = str(c.get("quote", "") or "").strip()
        if not chunk_id and not quote:
            continue
        if len(quote) > max_quote_chars:
            quote = quote[:max_quote_chars].rstrip() + "…"
        out_cites.append({"chunk_id": chunk_id, "quote": quote})
        if len(out_cites) >= max_cites:
            break

    return {"answer": ans, "citations": out_cites}
