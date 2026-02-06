from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenLocal:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Use dtype (torch_dtype is deprecated in newer transformers)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=dtype,
        )

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

        # Make sure pad token exists
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        # Tokenize on CPU first
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move to the SAME device as model
        dev = next(self.model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        do_sample = float(temperature) > 0.0

        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id),
        )
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)

        out = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Return completion only (same semantics as before)
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()
