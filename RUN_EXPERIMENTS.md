# Complete Guide: Running O-RAN RAG Experiments with Qwen3-4B

This guide provides all commands to run experiments on your full O-RAN specifications and get metric results.

## Prerequisites

```bash
cd /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3
```

### 1. Setup Environment

```bash
# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip setuptools wheel
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from transformers import AutoModelForCausalLM; print('Transformers OK')"
```

### 2. Prepare Data

```bash
# Copy full O-RAN specs to the required location
cp -r /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3/data/full_oran_specs/* \
    /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3/data/raw_specs/

# Verify PDFs are in place
ls -lh data/raw_specs/*.pdf | wc -l
```

---

## Full Experiment Pipeline

### Step 1: Build Corpus (Extract & Chunk PDFs)

This extracts text from PDFs and creates chunks aligned to section numbers.

```bash
bash scripts/01_build_corpus.sh
```

**Output:**
- `data/intermediate/chunks.jsonl` - All chunks with metadata
- `data/indexes/docstore.sqlite` - SQLite database of chunks

**Expected time:** 10-30 minutes (depends on PDF count and complexity)

### Step 2: Build Indexes (BM25 + FAISS Dense)

This creates both keyword and semantic search indexes.

```bash
bash scripts/02_build_index.sh
```

**Output:**
- `data/indexes/bm25/bm25.pkl` - BM25 index
- `data/indexes/faiss/faiss.index` - Dense embeddings index

**Expected time:** 5-20 minutes

### Step 3: Quick Test (Optional - Before Full Evaluation)

Test on a small sample first to verify everything works:

```bash
# Quick test on evaluation set
python scripts/mini_run_eval.py \
  --config configs/app.yaml \
  --eval data/eval/oran_eval.jsonl \
  --top_k 10 \
  --out_dir results/test_run_$(date +%Y%m%d_%H%M%S)

# Check results
cat results/test_run_*/mini_eval_*.summary.json | jq .
```

**Expected output:**
```json
{
  "total_n": 50,
  "micro_acc": 0.72,
  "micro_avg_citations": 2.3,
  "results": [...]
}
```

---

## Full Evaluation: Choose Your Setup

### Option A: Single GPU (Fastest for Testing)

```bash
# Run on single GPU (default)
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --top_k 10 \
  --out_dir data/reports/eval_single_gpu_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 \
  --log_prompt 0

# Results will be in: data/reports/eval_single_gpu_YYYYMMDD_HHMMSS/
```

**Expected time:** 1-4 hours (depending on benchmark size)

---

### Option B: Multi-GPU (Fastest for Production)

```bash
# Run on 4 GPUs (adjust GPU numbers for your setup)
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --gpus 0,1,2,3 \
  --top_k 10 \
  --out_dir data/reports/eval_mgpu_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 \
  --log_prompt 0

# Results will be in: data/reports/eval_mgpu_YYYYMMDD_HHMMSS/
```

**For 8 GPUs:**
```bash
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --gpus 0,1,2,3,4,5,6,7 \
  --top_k 10 \
  --out_dir data/reports/eval_mgpu_full_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 \
  --log_prompt 0

# Expected time: 30-60 minutes
```

---

### Option C: Using Provided Script

```bash
bash scripts/04_eval.sh
```

---

## Viewing Metric Results

### 1. After Single-GPU Evaluation

```bash
# View summary metrics
cat data/reports/eval_single_gpu_*/single.summary.json | jq .

# View per-question results (first 10)
head -10 data/reports/eval_single_gpu_*/*.jsonl | jq -c '{question, gold, pred_num, acc}'

# Calculate statistics
jq -r '.[] | select(.acc==1) | "Correct"' data/reports/eval_single_gpu_*/*.jsonl | wc -l
```

### 2. After Multi-GPU Evaluation

```bash
# View final aggregated results (MAIN METRICS)
cat data/reports/eval_mgpu_*/aggregate.summary.json | jq .

# View per-GPU results
cat data/reports/eval_mgpu_*/shard_*.summary.json | jq -c '{shard_id, total_n, micro_acc, micro_avg_citations}'

# View per-question with retrieval debug
jq -c '{question, gold, pred_num, acc, citations_count, _debug}' data/reports/eval_mgpu_*/*.jsonl | head -20
```

### 3. Extract Key Metrics

```bash
# Get overall accuracy and citations
jq '{total_n, micro_acc, micro_avg_citations}' data/reports/eval_mgpu_*/aggregate.summary.json

# Per-split breakdown (E=Easy, M=Medium, H=Hard)
jq '.results[] | {path, n, acc, avg_citations}' data/reports/eval_mgpu_*/shard_0_*.summary.json

# Failed predictions (acc == 0)
jq -c 'select(.acc==0) | {question, gold, pred_num}' data/reports/eval_mgpu_*/*.jsonl

# High-confidence answers (with citations)
jq -c 'select(.citations_count >= 2) | {question, pred_num, acc, citations_count}' data/reports/eval_mgpu_*/*.jsonl | head
```

---

## Understanding Output Structure

### Metric Meanings

| Metric | Description |
|--------|-------------|
| `total_n` | Total number of questions evaluated |
| `micro_acc` | Exact-match accuracy (% of correct answers) |
| `micro_avg_citations` | Average citations per answer |
| `acc` (per split) | Accuracy for Easy/Medium/Hard |
| `avg_citations` | Avg citations for that split |

### Output Files

For each evaluation run:

```
data/reports/eval_mgpu_YYYYMMDD_HHMMSS/
├── aggregate.summary.json           # FINAL METRICS (all GPUs combined)
├── shard_0_gpu0.jsonl               # Per-question results GPU0
├── shard_0_gpu0.summary.json        # GPU0 metrics
├── shard_0_gpu0.log.txt             # GPU0 detailed log
├── shard_1_gpu1.jsonl               # Per-question results GPU1
├── shard_1_gpu1.summary.json        # GPU1 metrics
├── shard_1_gpu1.log.txt             # GPU1 detailed log
└── ... (repeat for each GPU)
```

### JSON Structure of Results

**aggregate.summary.json:**
```json
{
  "total_n": 300,
  "micro_acc": 0.78,
  "micro_avg_citations": 2.4,
  "results": [
    {
      "path": "fin_E.json",
      "n": 100,
      "acc": 0.82,
      "avg_citations": 2.5
    },
    ...
  ]
}
```

**Per-question (*.jsonl, one line per question):**
```json
{
  "split": "fin_E",
  "question": "Which WG focuses on architecture?",
  "options": ["WG1", "WG2", "WG3"],
  "gold": "1",
  "pred_num": "1",
  "pred_text": "1",
  "acc": 1.0,
  "citations_count": 2,
  "citations": [
    {"chunk_id": "O-RAN.WG1.OAM::1.2::p5-6::b0001::0005::150::a1b2c3d4", "quote": "..."}
  ],
  "_debug": {
    "round": 1,
    "query": "Which WG focuses on architecture?",
    "fused_top": [[0.92, "chunk_id_1"], [0.88, "chunk_id_2"]],
    "reranked_top": [[0.89, "chunk_id_1"], [0.85, "chunk_id_2"]]
  }
}
```

---

## Complete End-to-End Script

Here's a single script that runs everything:

```bash
#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3"
cd "$PROJECT_DIR"

echo "[1/5] Activating environment..."
source .venv/bin/activate

echo "[2/5] Building corpus from PDFs..."
bash scripts/01_build_corpus.sh

echo "[3/5] Building indexes (BM25 + FAISS)..."
bash scripts/02_build_index.sh

echo "[4/5] Running quick test..."
python scripts/mini_run_eval.py \
  --config configs/app.yaml \
  --eval data/eval/oran_eval.jsonl \
  --out_dir results/test_$(date +%Y%m%d_%H%M%S)

echo "[5/5] Running full evaluation on 4 GPUs..."
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --gpus 0,1,2,3 \
  --out_dir data/reports/eval_full_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 \
  --log_prompt 0

echo "[SUCCESS] Evaluation complete!"
echo "Results saved to: data/reports/eval_full_*/"
echo "View metrics: cat data/reports/eval_full_*/aggregate.summary.json | jq ."
```

Save as `run_all.sh` and execute:
```bash
bash run_all.sh
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce max_new_tokens in configs/app.yaml
# Or reduce batch size for dense embedding generation
```

### Slow PDF Processing
```bash
# Reduce max_pages_per_pdf in configs/app.yaml
# default: 2000 (process all)
# try: 500 (first 500 pages per PDF)
```

### Specific GPU Problems
```bash
# Check available GPUs
nvidia-smi

# Run on specific GPU only
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m oran_rag.eval.run_eval --config configs/app.yaml --eval data/eval/oran_eval.jsonl
```

### Model Not Found
```bash
# Download Qwen3-4B model first
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')"
```

---

## Configuration Customization

Edit `configs/app.yaml` to tune:

```yaml
# LLM Settings
model:
  backend: local               # local or vllm
  name: Qwen/Qwen3-4B-Instruct-2507
  max_new_tokens: 512         # Reduce for speed, increase for verbosity
  temperature: 0.2            # Lower = more deterministic
  top_p: 0.9                  # Nucleus sampling

# Retrieval
retrieval:
  bm25_topk: 80               # Keyword search top-k
  dense_topk: 80              # Semantic search top-k
  fused_topk: 60              # After RRF fusion
  rerank_topk: 12             # After reranking
  enable_dense: true          # Enable semantic search
  enable_rerank: true         # Enable cross-encoder reranking

# Context Packing
packing:
  max_context_chars: 25000    # Max chars to pass to LLM
  neighbor_window: 1          # Include adjacent sections
```

---

## Key Commands Reference

| Task | Command |
|------|---------|
| Build corpus | `bash scripts/01_build_corpus.sh` |
| Build indexes | `bash scripts/02_build_index.sh` |
| Quick test | `python scripts/mini_run_eval.py --config configs/app.yaml --eval data/eval/oran_eval.jsonl` |
| Single GPU eval | `PYTHONPATH=src python -m oran_rag.eval.run_eval --config configs/app.yaml --bench_dir data/benchmarks/oranbench --splits E,M,H` |
| Multi-GPU eval | `PYTHONPATH=src python -m oran_rag.eval.run_eval --config configs/app.yaml --bench_dir data/benchmarks/oranbench --splits E,M,H --gpus 0,1,2,3` |
| View metrics | `jq . data/reports/eval_*/aggregate.summary.json` |
| View per-question | `jq -c '{question, gold, pred_num, acc}' data/reports/eval_*/*.jsonl \| head -20` |

