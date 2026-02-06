from __future__ import annotations
from typing import Any, Dict, List
from ..utils import read_jsonl

def load_eval(path: str) -> List[Dict[str, Any]]:

    return read_jsonl(path)
