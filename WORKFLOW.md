# O-RAN RAG Experiment Workflow & Quick Reference

## Visual Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    START: O-RAN RAG Experiments                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  1. PREPARE ENVIRONMENT         │
        │  .venv, pip install -r req.txt  │
        │  Expected: 5 min                │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  2. PREPARE DATA                │
        │  Copy PDFs to data/raw_specs/   │
        │  Expected: 1 min                │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  3. BUILD CORPUS                │
        │  Extract & chunk PDFs           │
        │  Expected: 15-45 min            │
        │  Output: chunks.jsonl           │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  4. BUILD INDEXES               │
        │  BM25 + FAISS Dense             │
        │  Expected: 10-25 min            │
        │  Output: *.pkl, *.index         │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────────────────────┐
        │  5. CHOOSE EVALUATION STRATEGY                  │
        └────┬──────────────────────────────┬─────────────┘
             │                              │
     ┌───────▼──────────┐          ┌────────▼──────────────┐
     │  OPTION A:       │          │  OPTION B:           │
     │  Single GPU      │          │  Multi-GPU (4-8)     │
     │  Expected: 2-6h  │          │  Expected: 30-120min │
     └───────┬──────────┘          └────────┬──────────────┘
             │                              │
     ┌───────▼──────────────────────────────▼──────────────┐
     │  RUN EVALUATION                                      │
     │  Bash: ./QUICK_START.sh                             │
     │  Or: PYTHONPATH=src python -m oran_rag.eval.run_eval│
     └───────┬──────────────────────────────────────────────┘
             │
        ┌────▼──────────────────────────────┐
        │  6. VIEW RESULTS                   │
        │  ./VIEW_RESULTS.sh                 │
        │  Or: jq . data/reports/eval_*     │
        └────┬──────────────────────────────┘
             │
        ┌────▼──────────────────────────────┐
        │  7. ANALYZE METRICS                │
        │  - Overall accuracy                │
        │  - Per-split performance           │
        │  - Citation quality                │
        │  - Failure analysis                │
        └────▼──────────────────────────────┘
             │
        ┌────▼──────────────────────────────┐
        │  END: Results in                   │
        │  data/reports/eval_mgpu_YYYYMMDD  │
        └────────────────────────────────────┘
```

## Three Ways to Run Experiments

### 1️⃣ FASTEST (Automated)
```bash
./QUICK_START.sh
# Runs all 6 steps automatically
# Time: Total pipeline ~2-3 hours
```

### 2️⃣ FLEXIBLE (Manual Steps)
```bash
bash scripts/01_build_corpus.sh
bash scripts/02_build_index.sh
PYTHONPATH=src python -m oran_rag.eval.run_eval --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench --splits E,M,H \
  --gpus 0,1,2,3 --out_dir data/reports/eval_$(date +%Y%m%d_%H%M%S)
```

### 3️⃣ CONTROLLED (Individual Tests)
```bash
# Test each component separately
bash scripts/01_build_corpus.sh      # Verify corpus building
bash scripts/02_build_index.sh       # Verify indexing
python scripts/mini_run_eval.py --eval data/eval/oran_eval.jsonl  # Validate
```

---

## The Exact Commands You Need

### One-Liner Setup
```bash
cd /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3 && \
python3.10 -m venv .venv && source .venv/bin/activate && \
pip install -U pip && pip install -r requirements.txt
```

### One-Liner Full Pipeline (with 4 GPUs)
```bash
cd /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3 && \
source .venv/bin/activate && \
mkdir -p data/raw_specs && \
cp /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3/data/full_oran_specs/*.pdf data/raw_specs/ && \
bash scripts/01_build_corpus.sh && \
bash scripts/02_build_index.sh && \
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --gpus 0,1,2,3 \
  --out_dir data/reports/eval_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 --log_prompt 0 && \
./VIEW_RESULTS.sh
```

---

## Configuration Profiles

### Profile 1: Fast (Demo/Testing)
```bash
# Edit configs/app.yaml
model:
  name: Qwen/Qwen3-4B-Instruct-2507
  max_new_tokens: 256
  temperature: 0.1

retrieval:
  enable_dense: false          # Skip dense retrieval
  enable_rerank: false         # Skip reranking
  bm25_topk: 30

ingest:
  max_pages_per_pdf: 100       # Reduce PDF pages
```

### Profile 2: Balanced (Recommended)
```bash
model:
  name: Qwen/Qwen3-4B-Instruct-2507
  max_new_tokens: 512
  temperature: 0.2

retrieval:
  enable_dense: true           # Enable dense
  enable_rerank: true          # Enable reranking
  bm25_topk: 80
  dense_topk: 80
  
ingest:
  max_pages_per_pdf: 2000      # Full docs
```

### Profile 3: High-Quality (Slower)
```bash
model:
  name: Qwen/Qwen3-4B-Instruct-2507
  max_new_tokens: 768
  temperature: 0.3

retrieval:
  enable_dense: true
  enable_rerank: true
  bm25_topk: 100
  dense_topk: 100
  fused_topk: 80
  
packing:
  max_context_chars: 35000     # Larger context
  neighbor_window: 2           # Include neighbors
```

---

## GPU Allocation Guide

| GPUs | Setup | Time | Best For |
|------|-------|------|----------|
| 1 | `--gpus 0` | 2-6h | Testing, development |
| 2 | `--gpus 0,1` | 1-3h | Small benchmarks |
| 4 | `--gpus 0,1,2,3` | 45-120m | Standard eval |
| 8 | `--gpus 0,1,2,3,4,5,6,7` | 30-60m | Production |

---

## Metrics Explained

### Overall Metrics (aggregate.summary.json)

```json
{
  "total_n": 300,              // Total questions evaluated
  "micro_acc": 0.72,           // Overall accuracy (72% correct)
  "micro_avg_citations": 2.3,  // Avg citations per answer
  "results": [                 // Per-split breakdown
    {
      "path": "fin_E.json",    // Easy split
      "n": 100,                // 100 questions
      "acc": 0.82,             // 82% accuracy on easy
      "avg_citations": 2.5     // 2.5 avg citations
    },
    ...
  ]
}
```

### Per-Question Results (*.jsonl)

```json
{
  "split": "fin_E",
  "question": "Which WG focuses on architecture?",
  "options": ["WG1", "WG2", "WG3"],
  "gold": "1",                          // Correct answer
  "pred_num": "1",                      // Predicted answer
  "acc": 1.0,                           // Correct (1.0) or Wrong (0.0)
  "citations_count": 2,                 // How many sources cited
  "citations": [                        // Actual citations
    {
      "chunk_id": "O-RAN.WG1.OAM::1.2::p5-6::b0001::0005::150::a1b2c3d4",
      "quote": "WG1 focuses on architecture..."
    }
  ]
}
```

---

## Common Analyses

### 1. Overall Performance
```bash
jq '{total: .total_n, accuracy: .micro_acc, citations: .micro_avg_citations}' \
  data/reports/eval_mgpu_*/aggregate.summary.json
```

### 2. Per-Difficulty Performance
```bash
jq '.results[] | {split: .path, n: .n, accuracy: (.acc * 100 | round), citations: .avg_citations}' \
  data/reports/eval_mgpu_*/aggregate.summary.json
```

### 3. Citation Quality
```bash
# Average citations per answer
jq -r '.citations_count' data/reports/eval_mgpu_*/*.jsonl | \
  awk '{sum+=$1; count++} END {print "Avg citations:", sum/count}'

# Questions with good citations (2+)
jq -c 'select(.citations_count >= 2)' data/reports/eval_mgpu_*/*.jsonl | wc -l
```

### 4. Failure Analysis
```bash
# What did the model get wrong?
jq -c 'select(.acc==0) | {q: .question, gold: .gold, pred: .pred_num}' \
  data/reports/eval_mgpu_*/*.jsonl | head -20
```

### 5. Retrieval Quality
```bash
# Do top-1 retrievals match reranking?
jq -c '._debug | {fused1: .fused_top[0][1], rerank1: .reranked_top[0][1]}' \
  data/reports/eval_mgpu_*/*.jsonl | head -10
```

---

## Expected Runtime

| Phase | Time |
|-------|------|
| Setup (venv + pip) | 10-15 min |
| Prepare data | 1 min |
| Build corpus | 15-45 min |
| Build indexes | 10-25 min |
| Quick test | 5-15 min |
| Full eval (1 GPU) | 2-6 hours |
| Full eval (4 GPUs) | 45-120 min |
| Full eval (8 GPUs) | 30-60 min |
| **Total pipeline (8 GPU)** | **~2 hours** |

---

## File Sizes Reference

| File | Size | Notes |
|------|------|-------|
| Full O-RAN specs PDFs | ~500MB | 100+ documents |
| chunks.jsonl | 50-200MB | Extracted chunks |
| docstore.sqlite | 100-300MB | Indexed chunks |
| bm25/bm25.pkl | 50-150MB | BM25 index |
| faiss/faiss.index | 200-500MB | Dense embeddings |
| eval results (\*.jsonl) | 50-500MB | Per-question results |

---

## Next Steps After Experiments

1. **Analyze Results**
   ```bash
   ./VIEW_RESULTS.sh
   jq . data/reports/eval_mgpu_*/aggregate.summary.json
   ```

2. **Identify Failure Patterns**
   ```bash
   jq -c 'select(.acc==0)' data/reports/eval_mgpu_*/*.jsonl > failures.jsonl
   ```

3. **Tune Configuration**
   - Edit `configs/app.yaml`
   - Re-run evaluation with new settings

4. **Deploy API Server**
   ```bash
   bash scripts/03_run_server_local.sh
   curl http://localhost:8000/docs
   ```

5. **Generate Report**
   ```bash
   python -c "
   import json
   with open('data/reports/eval_mgpu_*/aggregate.summary.json') as f:
       m = json.load(f)
   print(f'Accuracy: {m[\"micro_acc\"]*100:.1f}%')
   print(f'Avg Citations: {m[\"micro_avg_citations\"]:.2f}')
   "
   ```

---

## Support Commands

```bash
# Check CUDA/GPU status
nvidia-smi

# Monitor GPU during eval
nvidia-smi -l 1  # Update every 1 second

# Kill stuck process
pkill -f "oran_rag.eval"

# Clean up old results (keep last 3)
ls -d data/reports/eval_* | sort -r | tail -n +4 | xargs rm -rf

# Export results to CSV for analysis
jq -r '.results[] | [.path, .n, .acc, .avg_citations] | @csv' \
  data/reports/eval_*/aggregate.summary.json
```

