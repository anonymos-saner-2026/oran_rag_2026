# O-RAN RAG for Qwen3-4B (Spec-aware)

This project builds a **spec-aware RAG** system for **O-RAN Specifications** using **Qwen3-4B** (or any instruct LLM),
optimized for **small language models**:
- Chunking aligned to **section/clause numbering**
- **Hybrid retrieval** (BM25 + Dense embeddings) + RRF fusion
- Optional **reranking**
- **Section-aware context packing** (neighbor windows)
- Strict **JSON answer** with **citations**

## 0) Prerequisites
- Python 3.10+
- Your PDFs are **text selectable** (no OCR needed)
- (Recommended) GPU for Qwen3, otherwise use vLLM server on a GPU machine

## 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Put PDFs
Place O-RAN PDFs under:
```text
data/raw_specs/
```

## 3) Build corpus (chunks + docstore)
```bash
bash scripts/01_build_corpus.sh
```

Output:
- `data/intermediate/chunks.jsonl`
- `data/indexes/docstore.sqlite`

## 4) Build indexes (BM25 + Dense FAISS)
```bash
bash scripts/02_build_index.sh
```

## 5) Run API server
```bash
bash scripts/03_run_server_local.sh
```
Then open:
- http://localhost:8000/docs

## 6) Ask a question
```bash
curl -s http://localhost:8000/ask -H 'Content-Type: application/json' -d '{
  "question": "Which O-RAN Working Group focuses on the architecture description of Open Radio Access Networks?",
  "filters": {"wg": "WG1"},
  "top_k": 10
}' | jq
```

## Qwen backend options
In `configs/app.yaml`:
- `model.backend: local` uses `transformers` directly
- `model.backend: vllm` calls an OpenAI-compatible vLLM endpoint

If using vLLM:
1) Start vLLM with OpenAI-compatible server (example):
   ```bash
   python -m vllm.entrypoints.openai.api_server              --model Qwen/Qwen3-4B-Instruct-2507              --port 8001
   ```
2) Set env:
   ```bash
   export VLLM_BASE_URL=http://localhost:8001/v1
   export VLLM_API_KEY=EMPTY
   ```
3) Keep `model.backend: vllm` in config.

## Design choices for O-RAN specs
- Chunking by clause numbering is crucial for correct citations.
- Hybrid retrieval is important because specs contain many IDs, abbreviations, and exact tokens.
- Rerank + packing reduces noise for small LMs.

## Troubleshooting
- If PDF text extraction is messy, try different PDFs or reduce `max_pages_per_pdf` in config.
- If dense indexing is slow, reduce corpus size or switch embeddings model in config.
