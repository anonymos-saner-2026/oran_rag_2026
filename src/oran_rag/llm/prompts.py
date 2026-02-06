from __future__ import annotations
from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader, select_autoescape

class PromptRenderer:
    def __init__(self, prompts_dir: str):
        self.env = Environment(
            loader=FileSystemLoader(prompts_dir),
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, name: str, **kwargs: Any) -> str:
        t = self.env.get_template(name)
        return t.render(**kwargs)
