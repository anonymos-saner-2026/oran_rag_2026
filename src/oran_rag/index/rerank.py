from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # type: ignore

import torch


def get_search_text(c: Dict[str, Any]) -> str:
    title = str(c.get("title", "") or "")
    clause = str(c.get("clause_id", "") or "")
    text = str(c.get("text", "") or "")
    return f"{title}\n{clause}\n{text}".strip()


class Reranker:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers not installed or CrossEncoder unavailable.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(
        self,
        query: str,
        items: List[Tuple[float, Dict[str, Any]]],
        topk: int = 10,
        batch_size: int = 32,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        if topk <= 0:
            return []
        if not items:
            return []

        pairs = [(query, get_search_text(c)) for _, c in items]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        out = list(zip([float(s) for s in scores], [c for _, c in items]))
        out.sort(key=lambda x: x[0], reverse=True)
        return out[:topk]
