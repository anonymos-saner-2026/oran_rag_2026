from __future__ import annotations
from typing import Dict, Optional
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, str]] = None  # e.g., {"wg":"WG1", "version":"X.Y"}
    top_k: int = Field(default=10, ge=1, le=50)

class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, str]] = None
    top_k: int = Field(default=10, ge=1, le=50)
