from __future__ import annotations
import os
from typing import Any, Dict, Optional

try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError("vLLM not installed. Run: pip install vllm")

class VLLMClient:

    def __init__(
        self,
        model_name: str,
        base_url: str = None,
        api_key: str = None,
        gpu_memory_utilization: float = None,
    ):
        # For library mode, ignore base_url and api_key
        self.model_name = model_name
        llm_kwargs = {
            "model": model_name,
            "trust_remote_code": True,
        }
        if gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)
        # Initialize vLLM LLM
        self.llm = LLM(**llm_kwargs)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
