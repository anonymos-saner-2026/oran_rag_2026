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

## Running Evaluation with OpenRouter

You can run evaluation using **OpenRouter** as the LLM backend for any model available through their API (Qwen, Mistral, Llama, Claude, etc.).

### 1) Set up OpenRouter API key
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
export OPENROUTER_REFERER="https://yourwebsite.com"  # Optional but recommended
```

Or add to `.env`:
```
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_REFERER=https://yourwebsite.com
```

### 2) Create a config for OpenRouter (example: `configs/app_openrouter_qwen.yaml`)

```yaml
model:
  backend: openrouter
  name: Qwen/Qwen3-4B-Instruct-2507  # or qwen/qwen-32b, mistralai/mistral-7b, etc.
  base_url: https://openrouter.ai/api/v1
  # API key is read from OPENROUTER_API_KEY env var (see code)
  max_new_tokens: 512
  temperature: 0.25
  top_p: 0.9

retrieval:
  enable_dense: true
  dense_weight: 0.7
  enable_rerank: true
  
packing:
  max_context_chars: 16000
  neighbor_window: 3

ingest:
  max_pages_per_pdf: 2000
```

### 3) Run single-model evaluation on E subset (quick test)

```bash
python -m oran_rag.eval.run_eval \
  --config configs/app_openrouter_qwen.yaml \
  --eval data/benchmarks/oranbench/fin_E_10.jsonl \
  --out_dir data/reports/eval_openrouter_qwen3 \
  --top_k 10
```

### 4) Run full evaluation on all three splits (E, M, H)

```bash
python -m oran_rag.eval.run_eval \
  --config configs/app_openrouter_qwen.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --out_dir data/reports/eval_openrouter_qwen3_full \
  --top_k 10
```

### 5) Run multiple models in parallel on OpenRouter

Create a helper script `run_openrouter_models.sh`:

```bash
#!/bin/bash

# Models to evaluate
MODELS=(
  "qwen/qwen-4b"
  "mistralai/mistral-7b"
  "meta-llama/llama-2-7b"
)

SPLITS="E,M,H"
BENCH_DIR="data/benchmarks/oranbench"

for model in "${MODELS[@]}"; do
  model_name=$(echo "$model" | sed 's/\//_/g')
  out_dir="data/reports/eval_openrouter_${model_name}_$(date +%Y%m%d_%H%M%S)"
  
  echo "[START] Evaluating $model -> $out_dir"
  
  python -m oran_rag.eval.run_eval \
    --config configs/app.yaml \
    --bench_dir "$BENCH_DIR" \
    --splits "$SPLITS" \
    --out_dir "$out_dir" \
    --top_k 10 \
    --model_name "$model" &
done

wait
echo "[DONE] All evaluations completed"
```

Run it:
```bash
bash run_openrouter_models.sh
```

### 6) Check results and compare models

After evaluation completes, view the generated markdown report:
```bash
# View single-model report
cat data/reports/eval_openrouter_qwen3*/EXPERIMENT_REPORT.md

# View all CSV results (predictions vs. gold answers)
ls -1 data/reports/eval_openrouter_*/*.csv

# Compare metrics across models
for dir in data/reports/eval_openrouter_*/; do
  echo "=== $(basename $dir) ==="
  jq '{total_n, micro_acc, micro_avg_citations}' "$dir/aggregate.summary.json"
  echo ""
done
```

### 7) OpenRouter model selection tips

**Fast & affordable:**
- `qwen/qwen-4b` (cheapest)
- `mistralai/mistral-7b`

**Better quality but slower:**
- `meta-llama/llama-2-13b`
- `meta-llama/llama-2-70b`
- `mistralai/mixtral-8x7b`

**Best quality (if budget allows):**
- `anthropic/claude-3-sonnet`
- `openai/gpt-4`
- `openai/gpt-4-turbo`

Check [OpenRouter pricing](https://openrouter.ai/pricing) for costs per model.

### 8) Cost & rate limiting

- OpenRouter charges per token (input + output).
- Each evaluation sample generates ~200-500 tokens (varies by model).
- Budget estimates:
  - E subset (10q): ~$0.01–$0.10 depending on model
  - Full (400q): ~$1–$10 depending on model
- Rate limits: check your OpenRouter account dashboard.
- To stay within budget, test on smaller subsets first (`--eval fin_E_10.jsonl`).

### 9) Troubleshooting OpenRouter

**Authentication error:**
```bash
# Verify API key is set
echo $OPENROUTER_API_KEY

# Test connection
curl -X POST https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -d '{"model":"qwen/qwen-4b", "messages":[{"role":"user","content":"Hi"}]}'
```

**Model not found:**
- Check [OpenRouter available models](https://openrouter.ai/api/v1/models)
- Ensure model ID matches exactly (case-sensitive)

**Running out of budget:**
- Use cheaper models (Qwen-4B, Mistral-7B)
- Evaluate on smaller subsets (e.g., `fin_E_10.jsonl`)
- Increase token efficiency by reducing `max_context_chars` in config

### Output files

Each OpenRouter evaluation run produces:
- `EXPERIMENT_REPORT.md` – Full experiment summary with model, config, and metrics
- `fin_E.json.csv`, `fin_M.json.csv`, `fin_H.json.csv` – Per-split predictions vs. gold answers
- `aggregate.summary.json` – Overall accuracy and citation metrics
- `all_workers.log.txt` – Detailed logs with retrieval & LLM outputs
