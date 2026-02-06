from __future__ import annotations
from typing import Any, Dict, List

def exact_match(pred: str, gold: str) -> float:
    p = (pred or "").strip().lower()
    g = (gold or "").strip().lower()
    return 1.0 if p == g else 0.0

def citation_count(obj: Dict[str, Any]) -> int:
    cites = obj.get("citations", [])
    return len(cites) if isinstance(cites, list) else 0
