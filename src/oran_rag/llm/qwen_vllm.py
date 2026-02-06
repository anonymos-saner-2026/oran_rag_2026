from __future__ import annotations
import os
import time
import requests
from typing import Any, Dict, Optional

class QwenVLLM:

    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=120)
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After", "")
                    wait_s = 2.0 * (attempt + 1)
                    if retry_after.isdigit():
                        wait_s = max(wait_s, float(retry_after))
                    time.sleep(wait_s)
                    continue
                r.raise_for_status()
                data = r.json()
                return (data["choices"][0]["message"]["content"] or "").strip()
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))

        if last_err is not None:
            raise last_err
        raise RuntimeError("Request failed with unknown error")
