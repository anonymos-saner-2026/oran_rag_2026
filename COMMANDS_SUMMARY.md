# O-RAN RAG Experiments - Complete Command Reference

## Quick Start (Recommended)

Run everything with one command:

```bash
cd /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3

# Automatic pipeline with 4 GPUs
./QUICK_START.sh

# Or with 8 GPUs
GPUS=0,1,2,3,4,5,6,7 ./QUICK_START.sh

# View results
./VIEW_RESULTS.sh
```

---

## Manual Step-by-Step Commands

### Pre-requisites

```bash
cd /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3

# Create virtual environment (if not exists)
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

### Step 1: Prepare Full O-RAN Specs Data

```bash
# Copy full specs to raw_specs directory
mkdir -p data/raw_specs
cp /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3/data/full_oran_specs/*.pdf data/raw_specs/

# Verify (count PDFs)
ls data/raw_specs/*.pdf | wc -l
```

### Step 2: Build Corpus (Extract & Chunk PDFs)

```bash
# Method 1: Using shell script
bash scripts/01_build_corpus.sh

# Method 2: Direct Python command
PYTHONPATH=src python -m oran_rag.ingest.build_corpus --config configs/app.yaml
```

**Output:**
- `data/intermediate/chunks.jsonl` - Chunks with metadata
- `data/indexes/docstore.sqlite` - Searchable index
- **Expected time:** 15-45 minutes

### Step 3: Build Indexes (BM25 + FAISS)

```bash
# Method 1: Using shell script
bash scripts/02_build_index.sh

# Method 2: Direct Python command
PYTHONPATH=src python -m oran_rag.index.build --config configs/app.yaml
```

**Output:**
- `data/indexes/bm25/bm25.pkl` - BM25 keyword search
- `data/indexes/faiss/faiss.index` - Dense semantic search
- **Expected time:** 10-25 minutes

### Step 4: Quick Validation Test

```bash
# Test on small eval set before full benchmark
python scripts/mini_run_eval.py \
  --config configs/app.yaml \
  --eval data/eval/oran_eval.jsonl \
  --top_k 10 \
  --out_dir results/quick_test_$(date +%Y%m%d_%H%M%S)

# View quick results
cat results/quick_test_*/mini_eval_*.summary.json | jq .
```

**Expected time:** 5-20 minutes depending on eval set size

---

## Full Evaluation: Single vs Multi-GPU

### Option A: Single GPU Evaluation

```bash
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --top_k 10 \
  --out_dir data/reports/eval_single_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 \
  --log_prompt 0
```

**Expected time:** 2-6 hours

---

### Option B: Multi-GPU Evaluation (Recommended)

#### 4 GPUs (Common Setup)
```bash
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --gpus 0,1,2,3 \
  --top_k 10 \
  --out_dir data/reports/eval_mgpu_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 \
  --log_prompt 0
```

**Expected time:** 45-120 minutes

#### 8 GPUs (Faster)
```bash
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --gpus 0,1,2,3,4,5,6,7 \
  --top_k 10 \
  --out_dir data/reports/eval_mgpu_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 \
  --log_prompt 0
```

**Expected time:** 30-60 minutes

#### Specific GPUs (e.g., non-consecutive)
```bash
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --gpus 0,2,4,6 \
  --top_k 10 \
  --out_dir data/reports/eval_mgpu_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 \
  --log_prompt 0
```

---

## Viewing & Analyzing Results

### Quick View (Automatic)
```bash
# View latest results automatically
./VIEW_RESULTS.sh

# View specific results directory
./VIEW_RESULTS.sh data/reports/eval_mgpu_20260204_120000
```

### Manual Result Analysis

#### View Overall Metrics
```bash
# Multi-GPU results
jq . data/reports/eval_mgpu_*/aggregate.summary.json

# Single-GPU results
jq . data/reports/eval_single_*/single.summary.json
```

#### View Per-Split Breakdown
```bash
jq '.results[] | {split: .path, questions: .n, accuracy: .acc, citations: .avg_citations}' \
  data/reports/eval_mgpu_*/aggregate.summary.json
```

#### View Per-Question Results
```bash
# First 10 questions
jq -c '{q: .question, gold: .gold, pred: .pred_num, acc: .acc}' data/reports/eval_mgpu_*/*.jsonl | head -10

# All questions with accuracy
jq -c '{q: .question, gold: .gold, pred: .pred_num, acc: .acc}' data/reports/eval_mgpu_*/*.jsonl

# Only correct answers
jq -c 'select(.acc==1)' data/reports/eval_mgpu_*/*.jsonl

# Only incorrect answers
jq -c 'select(.acc==0)' data/reports/eval_mgpu_*/*.jsonl
```

#### View Citations
```bash
# Questions with citations
jq -c '{q: .question, pred: .pred_num, citations_count: .citations_count}' data/reports/eval_mgpu_*/*.jsonl | head -10

# Show actual citations
jq -c 'select(.citations_count > 0) | {q: .question, cites: .citations[0:2]}' data/reports/eval_mgpu_*/*.jsonl | head -5
```

#### View Retrieval Debug Info
```bash
# Show fused vs reranked rankings
jq -c '{q: .question, fused_top: ._debug.fused_top[0:3], reranked_top: ._debug.reranked_top[0:3]}' \
  data/reports/eval_mgpu_*/*.jsonl | head -5
```

#### Calculate Statistics
```bash
# Accuracy percentage
jq -r '.acc' data/reports/eval_mgpu_*/*.jsonl | awk '{sum+=$1; count++} END {printf "Accuracy: %.1f%% (%d/%d)\n", sum*100/count, sum, count}'

# Average citations
jq -r '.citations_count' data/reports/eval_mgpu_*/*.jsonl | awk '{sum+=$1; count++} END {printf "Avg citations: %.2f\n", sum/count}'

# Failure analysis
echo "=== FAILURE ANALYSIS ===" && \
jq -c 'select(.acc==0) | {split: .split, q: .question, gold: .gold, pred: .pred_num}' data/reports/eval_mgpu_*/*.jsonl | \
  jq -s 'group_by(.split) | map({split: .[0].split, count: length})'
```

---

## Configuration Tuning

Edit `configs/app.yaml` for different configurations:

### Option 1: Faster Inference (Lower Quality)
```yaml
model:
  max_new_tokens: 256        # Reduced from 512
  temperature: 0.1           # More deterministic
  
retrieval:
  bm25_topk: 40              # Reduced from 80
  dense_topk: 40             # Reduced from 80
  fused_topk: 30             # Reduced from 60
  enable_rerank: false       # Disable reranking to speed up
  
packing:
  max_context_chars: 15000   # Reduced from 25000
  neighbor_window: 0         # No neighbor sections
```

### Option 2: Better Quality (Slower Inference)
```yaml
model:
  max_new_tokens: 768        # Increased from 512
  temperature: 0.3           # More diverse
  
retrieval:
  bm25_topk: 100             # Increased from 80
  dense_topk: 100            # Increased from 80
  fused_topk: 80             # Increased from 60
  enable_rerank: true        # Keep reranking
  
packing:
  max_context_chars: 35000   # Increased from 25000
  neighbor_window: 2         # Include 2 neighbor sections
```

---

## Full End-to-End Example

Complete pipeline from start to results:

```bash
#!/bin/bash
cd /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3

echo "Starting O-RAN RAG Experiments..."
source .venv/bin/activate

echo "[1/6] Preparing data..."
mkdir -p data/raw_specs
cp /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3/data/full_oran_specs/*.pdf data/raw_specs/

echo "[2/6] Building corpus..."
PYTHONPATH=src python -m oran_rag.ingest.build_corpus --config configs/app.yaml

echo "[3/6] Building indexes..."
PYTHONPATH=src python -m oran_rag.index.build --config configs/app.yaml

echo "[4/6] Quick validation test..."
python scripts/mini_run_eval.py --config configs/app.yaml --eval data/eval/oran_eval.jsonl

echo "[5/6] Full evaluation (4 GPUs)..."
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --gpus 0,1,2,3 \
  --out_dir data/reports/eval_full_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 --log_prompt 0

echo "[6/6] Displaying results..."
./VIEW_RESULTS.sh

echo "✓ Complete!"
```

Save as `run_pipeline.sh` and execute:
```bash
bash run_pipeline.sh
```

---

## Environment Variables

```bash
# Set specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run specific number of GPUs
GPUS=0,1,2,3 ./QUICK_START.sh

# Reduce GPU memory (for OOM errors)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Disable dense retrieval (faster, lower quality)
# Edit configs/app.yaml: enable_dense: false
```

---

## Troubleshooting Commands

```bash
# Check GPU availability
nvidia-smi

# Check model download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')"

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name()}')"

# Test single GPU only
CUDA_VISIBLE_DEVICES=0 bash scripts/02_build_index.sh

# Run with verbose logging
PYTHONPATH=src python -m oran_rag.eval.run_eval --config configs/app.yaml --eval data/eval/oran_eval.jsonl --log_prompt 1 --log_raw 1
```

---

## Expected Results Range

Based on O-RAN specifications complexity:

| Metric | Typical Range |
|--------|---------------|
| Accuracy (Easy) | 75-85% |
| Accuracy (Medium) | 60-75% |
| Accuracy (Hard) | 40-60% |
| Avg Citations | 2.0-3.0 |
| Overall Micro-Accuracy | 60-75% |

---

## Output File Locations

```
data/
├── raw_specs/                          # Input PDFs
├── intermediate/
│   └── chunks.jsonl                    # Extracted chunks
├── indexes/
│   ├── docstore.sqlite                 # Chunk database
│   ├── bm25/                           # BM25 index
│   └── faiss/                          # Dense index
└── benchmarks/
    └── oranbench/
        ├── fin_E.json                  # Easy split
        ├── fin_M.json                  # Medium split
        └── fin_H.json                  # Hard split

results/
└── test_*/                             # Quick test results

data/reports/
├── eval_single_*/                      # Single-GPU results
│   └── single.summary.json
├── eval_mgpu_*/                        # Multi-GPU results
│   ├── aggregate.summary.json          # MAIN METRICS
│   ├── shard_0_gpu0.jsonl              # Per-question GPU0
│   ├── shard_0_gpu0.summary.json       # GPU0 metrics
│   ├── shard_0_gpu0.log.txt            # GPU0 logs
│   ├── shard_1_gpu1.*                  # ... GPU1
│   └── ...
```

---

## Quick Reference Table

| Task | Command |
|------|---------|
| **Setup** | `python3.10 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |
| **Prepare data** | `cp data/full_oran_specs/*.pdf data/raw_specs/` |
| **Build corpus** | `bash scripts/01_build_corpus.sh` |
| **Build indexes** | `bash scripts/02_build_index.sh` |
| **Quick test** | `python scripts/mini_run_eval.py --config configs/app.yaml --eval data/eval/oran_eval.jsonl` |
| **Full eval (4 GPU)** | `PYTHONPATH=src python -m oran_rag.eval.run_eval --config configs/app.yaml --bench_dir data/benchmarks/oranbench --splits E,M,H --gpus 0,1,2,3` |
| **Full eval (8 GPU)** | `PYTHONPATH=src python -m oran_rag.eval.run_eval --config configs/app.yaml --bench_dir data/benchmarks/oranbench --splits E,M,H --gpus 0,1,2,3,4,5,6,7` |
| **View results** | `./VIEW_RESULTS.sh` |
| **View metrics** | `jq . data/reports/eval_mgpu_*/aggregate.summary.json` |
| **View questions** | `jq -c '{q: .question, gold: .gold, pred: .pred_num, acc: .acc}' data/reports/eval_mgpu_*/*.jsonl` |

