from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

from ..utils import load_yaml
from ..index.bm25_index import load_index as load_bm25, search as bm25_search
from ..index.dense_index import load_index as load_dense, search as dense_search
from ..index.hybrid import rrf_fuse
from ..index.rerank import Reranker
from ..llm.prompts import PromptRenderer
import os
from ..llm.qwen_local import QwenLocal
from ..llm.vllm_client import VLLMClient
from ..llm.gemma2_local import Gemma2Local
from ..llm.qwen_vllm import QwenVLLM
from .packer import fetch_neighbors, trim_contexts_by_chars, to_prompt_context
from .gate import need_fallback
from .citations import safe_json_loads, clamp_citations


# -----------------------------
# Rewrite guards
# -----------------------------
_RE_ANSWER_PREFIX = re.compile(r"^\s*(final\s*:|answer\s*:|option\s*:)\s*", re.IGNORECASE)
_RE_JUST_OPTION = re.compile(r"^\s*\(?\s*([1-9]|10)\s*\)?\s*$")
_RE_OPTION_LIKE = re.compile(r"^\s*(answer|final|option)\s*[:：]?\s*\(?\s*([1-9]|10)\s*\)?\s*$", re.IGNORECASE)
_RE_TOO_MUCH_PROMPT_ECHO = re.compile(r"USER QUESTION:|Output ONLY|Rewrite the user's question", re.IGNORECASE)

# If rewrite is very short or looks like “Answer: 2”, reject.
_MIN_REWRITE_CHARS = 12


def _clean_one_line(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # take first non-empty line
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    s = lines[0]
    s = _RE_ANSWER_PREFIX.sub("", s).strip()
    # remove surrounding quotes
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ["'", '"']):
        s = s[1:-1].strip()
    return s


def _looks_like_answer(s: str) -> bool:
    if not s:
        return True
    if _RE_JUST_OPTION.match(s):
        return True
    if _RE_OPTION_LIKE.match(s):
        return True
    # common failure: "Answer: 3" / "2"
    if s.lower().startswith(("answer", "final", "option")) and any(ch.isdigit() for ch in s):
        return True
    return False


def _is_bad_rewrite(candidate: str, original: str) -> Tuple[bool, str]:
    """
    Return (bad?, reason)
    """
    c = (candidate or "").strip()
    if not c:
        return True, "empty"
    if _RE_TOO_MUCH_PROMPT_ECHO.search(c):
        return True, "prompt_echo"
    if _looks_like_answer(c):
        return True, "looks_like_answer"
    if len(c) < _MIN_REWRITE_CHARS:
        return True, "too_short"
    # If it removed almost everything / or is identical, it's not harmful but no benefit.
    # We'll allow identical (treated as no-op).
    return False, ""


@dataclass
class RAGEngine:
    cfg: Dict[str, Any]
    bm25: Dict[str, Any]
    dense: Optional[Dict[str, Any]]
    reranker: Optional[Reranker]
    llm: Any
    prompts: PromptRenderer

    @classmethod
    def from_config(cls, config_path: str) -> "RAGEngine":
        cfg = load_yaml(config_path)

        bm25 = load_bm25(cfg["paths"]["bm25_dir"])

        dense = None
        if cfg["retrieval"].get("enable_dense", True):
            try:
                dense = load_dense(cfg["paths"]["faiss_dir"])
            except Exception as e:
                print(f"[WARN] Dense index not loaded: {e}. Proceeding without dense.")

        reranker = None
        if cfg["retrieval"].get("enable_rerank", True):
            try:
                # allow optional device/max_length in config
                rr_cfg = cfg.get("retrieval", {})
                rr_device = rr_cfg.get("reranker_device", None)
                rr_maxlen = int(rr_cfg.get("reranker_max_length", 512))
                reranker = Reranker(rr_cfg["reranker_model"], device=rr_device, max_length=rr_maxlen)
            except Exception as e:
                print(f"[WARN] Reranker not available: {e}. Proceeding without rerank.")

        backend = cfg["model"]["backend"]
        model_name = cfg["model"]["name"]
        if backend == "local":
            if "gemma" in model_name.lower():
                llm = Gemma2Local(model_name)
            else:
                llm = QwenLocal(model_name)
        elif backend == "vllm":
            gpu_mem = cfg["model"].get("gpu_memory_utilization", None)
            llm = VLLMClient(cfg["model"]["name"], gpu_memory_utilization=gpu_mem)
        elif backend in ("remote", "vllm_api", "openai"):
            base_url = cfg["model"].get("base_url", "").strip()
            api_key_env = cfg["model"].get("api_key_env", "OPENAI_API_KEY")
            api_key = "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"
            if not base_url:
                raise ValueError("model.base_url is required for remote backend")
            if not api_key:
                raise ValueError(f"API key missing. Set env var: {api_key_env}")
            llm = QwenVLLM(cfg["model"]["name"], base_url=base_url, api_key=api_key)
        else:
            raise ValueError(f"Unknown model.backend: {backend}")

        prompts = PromptRenderer("configs/prompts")
        return cls(cfg=cfg, bm25=bm25, dense=dense, reranker=reranker, llm=llm, prompts=prompts)

    def retrieve(self, question: str, filters: Optional[Dict[str, str]] = None) -> List[Tuple[float, Dict[str, Any]]]:
        r_cfg = self.cfg["retrieval"]
        a = bm25_search(self.bm25, question, topk=int(r_cfg["bm25_topk"]), filters=filters)

        if self.dense is not None:
            b = dense_search(self.dense, question, topk=int(r_cfg["dense_topk"]), filters=filters)
        else:
            b = []

        # optional RRF knobs
        w_a = float(r_cfg.get("rrf_w_bm25", 1.0))
        w_b = float(r_cfg.get("rrf_w_dense", 1.0))
        cap_a = int(r_cfg.get("rrf_cap_bm25", r_cfg.get("bm25_topk", 80)))
        cap_b = int(r_cfg.get("rrf_cap_dense", r_cfg.get("dense_topk", 80)))

        fused = rrf_fuse(
            a, b,
            k=int(r_cfg["rrf_k"]),
            topk=int(r_cfg["fused_topk"]),
            w_a=w_a, w_b=w_b,
            cap_a=cap_a, cap_b=cap_b,
        )
        return fused

    def rerank(self, question: str, items: List[Tuple[float, Dict[str, Any]]]) -> List[Tuple[float, Dict[str, Any]]]:
        topk = int(self.cfg["retrieval"]["rerank_topk"])
        if self.reranker is None:
            return items[:topk]
        batch_size = int(self.cfg["retrieval"].get("reranker_batch_size", 32))
        return self.reranker.rerank(question, items, topk=topk, batch_size=batch_size)

    def rewrite_query(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (rewrite_or_original, debug_dict)
        """
        dbg: Dict[str, Any] = {"enabled": True, "raw": None, "clean": None, "used": False, "reject_reason": None}

        p = self.prompts.render("query_rewrite.jinja", question=question)
        out = self.llm.generate(
            p,
            max_new_tokens=64,
            temperature=0.0,  # rewrite should be deterministic
            top_p=1.0,
        )
        dbg["raw"] = out

        cand = _clean_one_line(out)
        dbg["clean"] = cand

        bad, reason = _is_bad_rewrite(cand, question)
        if bad:
            dbg["used"] = False
            dbg["reject_reason"] = reason
            return question, dbg

        # if identical, treat as no-op but still ok
        if cand.strip() == question.strip():
            dbg["used"] = False
            dbg["reject_reason"] = "same_as_original"
            return question, dbg

        dbg["used"] = True
        return cand, dbg

    def answer(
        self,
        question: str,
        filters: Optional[Dict[str, str]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        gate_cfg = self.cfg["gate"]
        pack_cfg = self.cfg["packing"]
        docstore = self.cfg["paths"]["docstore_path"]

        rounds = 0
        cur_q = question
        last_debug: Dict[str, Any] = {}
        rewrite_dbg: Optional[Dict[str, Any]] = None

        while True:
            rounds += 1
            fused = self.retrieve(cur_q, filters=filters)
            reranked = self.rerank(cur_q, fused)

            last_debug = {
                "round": rounds,
                "query": cur_q,
                "rewrite": rewrite_dbg,
                "fused_top": [(float(s), c["chunk_id"]) for s, c in fused[:5]],
                "reranked_top": [(float(s), c["chunk_id"]) for s, c in reranked[:5]],
                "reranked_scores": [float(s) for s, _ in reranked[:5]],
            }

            if not need_fallback(reranked, float(gate_cfg["min_rerank_score"])):
                break
            if rounds >= int(gate_cfg["max_rounds"]):
                break

            if gate_cfg.get("enable_query_rewrite", True):
                # IMPORTANT: rewrite from ORIGINAL question (not from cur_q)
                cur_q, rewrite_dbg = self.rewrite_query(question)
            else:
                break

        seed_chunks = [c for _, c in reranked[:max(3, int(top_k))]]
        contexts = fetch_neighbors(
            db_path=docstore,
            seed_chunks=seed_chunks,
            neighbor_window=int(pack_cfg["neighbor_window"]),
        )
        contexts = trim_contexts_by_chars(contexts, max_chars=int(pack_cfg["max_context_chars"]))
        prompt_contexts = to_prompt_context(contexts)

        prompt = self.prompts.render(
            "answer_with_citations.jinja",
            question=question,
            contexts=prompt_contexts,
        )
        raw = self.llm.generate(
            prompt,
            max_new_tokens=int(self.cfg["model"]["max_new_tokens"]),
            temperature=float(self.cfg["model"]["temperature"]),
            top_p=float(self.cfg["model"]["top_p"]),
        )
        obj = safe_json_loads(raw)
        obj = clamp_citations(obj, max_cites=3)
        obj["_debug"] = last_debug
        obj["_raw"] = raw  # <-- để bạn log full LLM output ra txt/jsonl khi eval
        return obj
