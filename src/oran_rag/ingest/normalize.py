from __future__ import annotations
import re

def cleanup_common_artifacts(text: str) -> str:
    # Remove repeated spaces and fix hyphen line breaks if they appear
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
