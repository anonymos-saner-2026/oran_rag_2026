from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

WORD_RE = re.compile(r"[A-Za-z0-9_/\-\.]+")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    # handle path without directory (e.g., "chunks.jsonl")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def guess_wg_from_filename(name: str) -> str:
    """
    Detect WG number robustly even when surrounded by underscores/hyphens, e.g.:
      "..._WG3_..." , "...-WG4-..." , "O-RAN.WG1...."
    Avoid using \\b because '_' is a word character in regex word-boundary logic.
    """
    # Pattern: boundary (non-alnum or start) + WG + optional spaces/dot + digits + boundary (non-alnum or end)
    m = re.search(r"(?:^|[^0-9A-Za-z])WG[\s\.]*([0-9]{1,2})(?:[^0-9A-Za-z]|$)", name, flags=re.IGNORECASE)
    if m:
        return f"WG{m.group(1)}"

    # Also allow "O-RAN.WG3" where boundary before might be '.' which is covered above,
    # but keep a fallback for weird cases.
    m = re.search(r"WG[\s\.]*([0-9]{1,2})", name, flags=re.IGNORECASE)
    if m:
        return f"WG{m.group(1)}"

    return "UNKNOWN"


def guess_version_from_filename(name: str) -> str:
    # Prefer pattern with explicit 'v' (case-insensitive), allow '_' or other non-alnum before v
    m = re.search(r"(?:^|[^0-9A-Za-z])v(\d+\.\d+(?:\.\d+)*)", name, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # fallback: first x.y(.z) occurrence with non-alnum boundary
    m = re.search(r"(?:^|[^0-9A-Za-z])(\d+\.\d+(?:\.\d+)*)", name)
    if m:
        return m.group(1)
    return "UNKNOWN"


def is_normative_text(text: str) -> bool:
    # normative language in specs
    low = (text or "").lower()
    return any(k in low for k in [" shall ", " should ", " must ", " may "])
def is_definition_section(text: str) -> bool:
    """
    Heuristic to tag definition-style chunks in telecom/spec documents.
    Works for ETSI/O-RAN specs where definitions appear under:
      - "Definitions", "Terms and definitions"
      - "Abbreviations", "Acronyms"
      - list-like "Term – definition" or "Term: definition"
    """
    t = (text or "").strip()
    if not t:
        return False

    low = " " + t.lower() + " "

    # common section markers
    if any(k in low for k in [
        " terms and definitions ",
        " term and definitions ",
        " definitions ",
        " abbreviations ",
        " acronym ",
        " acronyms ",
    ]):
        return True

    # looks like a glossary entry line: "Term: ..." or "Term – ..." near the beginning
    # We check only first ~400 chars to avoid false positives in long paragraphs.
    head = t[:400]
    if re.search(r"^[A-Z][A-Za-z0-9/\-\(\)\s]{1,50}\s*[:\-–]\s+\S+", head):
        return True

    # looks like enumerated definition style:
    # "3.1 Term" then following text starts with "is/are/means/refers"
    if re.search(r"^\s*\d+(\.\d+){0,3}\s+[A-Z][A-Za-z0-9/\-\(\)\s]{1,60}\s*$", head.splitlines()[0] if head.splitlines() else head):
        if re.search(r"\b(is|are|means|refers to|shall mean)\b", low):
            return True

    return False
