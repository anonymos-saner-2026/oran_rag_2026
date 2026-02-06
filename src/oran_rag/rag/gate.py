from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

def need_fallback(reranked: List[Tuple[float, Dict[str, Any]]], min_score: float) -> bool:
    if not reranked:
        return True
    best = reranked[0][0]
    return best < min_score

def definition_first_filters(filters: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    # Placeholder: we filter later by is_definition in scoring, not here.
    return filters or {}
