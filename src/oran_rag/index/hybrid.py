from __future__ import annotations

from typing import Any, Dict, List, Tuple


def rrf_fuse(
    a: List[Tuple[float, Dict[str, Any]]],
    b: List[Tuple[float, Dict[str, Any]]],
    k: int = 60,
    topk: int = 50,
    # NEW: control noisy channel contribution
    w_a: float = 1.0,
    w_b: float = 1.0,
    cap_a: int = 100,
    cap_b: int = 100,
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Reciprocal Rank Fusion.

    Inputs a, b should already be sorted desc by their own scores.
    Score = sum( weight / (k + rank) ).

    Added:
    - cap_a/cap_b: only take top-N from each list to reduce noise
    - w_a/w_b: weight contributions
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    def add(lst: List[Tuple[float, Dict[str, Any]]], w: float, cap: int) -> None:
        for r, (_, c) in enumerate(lst[: max(0, cap)], start=1):
            cid = c["chunk_id"]
            chunk_map[cid] = c
            scores[cid] = scores.get(cid, 0.0) + float(w) * (1.0 / (k + r))

    add(a, w=w_a, cap=cap_a)
    add(b, w=w_b, cap=cap_b)

    items = [(s, chunk_map[cid]) for cid, s in scores.items()]
    items.sort(key=lambda x: x[0], reverse=True)
    return items[:topk]
