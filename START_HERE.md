# 🚀 QUICK START - Copy & Paste Commands

> **Most important:** Copy these commands and run them in order

## Prerequisites (5 min)

```bash
cd /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3

# Setup environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## Automatic Pipeline (Recommended - 2 hours)

### One Command That Does Everything:

```bash
cd /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3
./QUICK_START.sh
```

That's it! This will:
1. ✅ Copy full O-RAN specs
2. ✅ Build corpus from PDFs
3. ✅ Create indexes
4. ✅ Run evaluation on 4 GPUs
5. ✅ Show you the results

---

## Alternative: Manual Steps (More Control)

### Step 1: Prepare Data (1 min)
```bash
cd /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3
source .venv/bin/activate

mkdir -p data/raw_specs
cp /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3/data/full_oran_specs/*.pdf data/raw_specs/

# Verify
ls data/raw_specs/*.pdf | wc -l
```

### Step 2: Build Corpus (20-30 min)
```bash
bash scripts/01_build_corpus.sh
```

### Step 3: Build Indexes (10-20 min)
```bash
bash scripts/02_build_index.sh
```

### Step 4: Run Evaluation (1-2 hours)

**Option A: 4 GPUs (Recommended)**
```bash
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --gpus 0,1,2,3 \
  --out_dir data/reports/eval_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 --log_prompt 0
```

**Option B: 8 GPUs (Faster)**
```bash
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --gpus 0,1,2,3,4,5,6,7 \
  --out_dir data/reports/eval_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 --log_prompt 0
```

**Option C: Single GPU (Slowest)**
```bash
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --out_dir data/reports/eval_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 --log_prompt 0
```

---

## View Results

### Automatic Results Viewer:
```bash
./VIEW_RESULTS.sh
```

### Manual Viewing:

**View Overall Metrics:**
```bash
jq . data/reports/eval_mgpu_*/aggregate.summary.json
```

**View Accuracy Breakdown:**
```bash
jq '.results[] | {split: .path, accuracy: .acc, citations: .avg_citations}' \
  data/reports/eval_mgpu_*/aggregate.summary.json
```

**View Individual Questions:**
```bash
jq -c '{question, gold, pred: .pred_num, acc}' data/reports/eval_mgpu_*/*.jsonl | head -20
```

**View Failed Predictions:**
```bash
jq -c 'select(.acc==0) | {question, gold, pred: .pred_num}' data/reports/eval_mgpu_*/*.jsonl
```

---

## Expected Results

After evaluation completes, you'll see:

```
==================================================
✓ EXPERIMENTS COMPLETE!
==================================================

Results Directory: data/reports/eval_oran_qwen3_YYYYMMDD_HHMMSS

METRICS SUMMARY:
{
  "total_n": 300,
  "micro_acc": 0.72,
  "micro_avg_citations": 2.4,
  "results": [...]
}

QUICK ANALYSIS:
  - Total questions: 300
  - Overall accuracy: 72%
  - Avg citations per answer: 2.4
```

---

## Configuration Changes (Optional)

### For Faster Results (Lower Quality):
Edit `configs/app.yaml`:
```yaml
model:
  max_new_tokens: 256        # was 512
  
retrieval:
  enable_rerank: false       # disable reranking
  
ingest:
  max_pages_per_pdf: 500     # limit pages per PDF
```

### For Better Results (Slower):
```yaml
model:
  max_new_tokens: 768        # was 512
  
retrieval:
  enable_rerank: true        # keep reranking
  bm25_topk: 100             # was 80
  dense_topk: 100            # was 80
  
packing:
  max_context_chars: 35000   # was 25000
  neighbor_window: 2         # was 1
```

---

## Troubleshooting

### CUDA Out of Memory?
```bash
# Edit configs/app.yaml:
model:
  max_new_tokens: 256  # reduce from 512
```

### GPUs Not Detected?
```bash
# Check GPU availability:
nvidia-smi

# Run on specific GPU only:
CUDA_VISIBLE_DEVICES=0 bash scripts/02_build_index.sh
```

### Process Stuck?
```bash
# Kill all oran_rag processes:
pkill -f "oran_rag"
```

### Model Download Issue?
```bash
# Download Qwen3-4B first:
python -c "from transformers import AutoModelForCausalLM; \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')"
```

---

## File Locations

```
📁 Project Root:
   /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/
   tungnvt5_node3/oran_rag/oran-rag-qwen3

📁 Input Data:
   data/raw_specs/          # Your PDFs go here
   data/full_oran_specs/    # Original O-RAN specs

📁 Generated Data:
   data/intermediate/chunks.jsonl    # Extracted chunks
   data/indexes/docstore.sqlite      # Chunk database
   data/indexes/bm25/                # BM25 index
   data/indexes/faiss/               # Dense embeddings

📁 Evaluation Results:
   data/reports/eval_mgpu_YYYYMMDD_HHMMSS/
   ├── aggregate.summary.json        # 📊 MAIN METRICS
   ├── shard_0_gpu0.jsonl            # Results GPU 0
   ├── shard_0_gpu0.log.txt          # Logs GPU 0
   └── ...

🔧 Configuration:
   configs/app.yaml                  # Main config file
   configs/prompts/                  # LLM prompts

📄 Helper Scripts:
   QUICK_START.sh                    # Run all in one
   VIEW_RESULTS.sh                   # View results
   COMMANDS_SUMMARY.md               # Full command reference
```

---

## Key Commands Cheat Sheet

| What | Command |
|------|---------|
| **Run Everything** | `./QUICK_START.sh` |
| **Build Corpus** | `bash scripts/01_build_corpus.sh` |
| **Build Indexes** | `bash scripts/02_build_index.sh` |
| **Evaluate (4 GPU)** | `PYTHONPATH=src python -m oran_rag.eval.run_eval --config configs/app.yaml --bench_dir data/benchmarks/oranbench --splits E,M,H --gpus 0,1,2,3` |
| **View Results** | `./VIEW_RESULTS.sh` |
| **See Metrics** | `jq . data/reports/eval_mgpu_*/aggregate.summary.json` |
| **See Questions** | `jq -c '{q: .question, gold, pred: .pred_num, acc}' data/reports/eval_mgpu_*/*.jsonl \| head -20` |

---

## Timeline

- **Setup**: 10 min
- **Prepare Data**: 1 min  
- **Build Corpus**: 20-30 min
- **Build Indexes**: 10-20 min
- **Evaluation (4 GPU)**: 1-2 hours
- **View Results**: < 1 min
- **TOTAL**: ~2-3 hours ✅

---

## What Happens After Evaluation?

You'll get a **metrics report** with:

```json
{
  "total_n": 300,                    // Total questions
  "micro_acc": 0.72,                 // 72% accuracy ✓
  "micro_avg_citations": 2.4,        // Average citations ✓
  "results": [
    {
      "path": "fin_E.json",          // Easy questions
      "n": 100,
      "acc": 0.82,                   // 82% on easy
      "avg_citations": 2.5
    },
    {
      "path": "fin_M.json",          // Medium questions
      "n": 100,
      "acc": 0.72,                   // 72% on medium
      "avg_citations": 2.4
    },
    {
      "path": "fin_H.json",          // Hard questions
      "n": 100,
      "acc": 0.62,                   // 62% on hard
      "avg_citations": 2.3
    }
  ]
}
```

Plus individual question results with:
- ✓ Predicted answers
- ✓ Citations (with source chunks)
- ✓ Retrieval debug info
- ✓ Model reasoning

---

## Questions?

Check these files for more details:
- `WORKFLOW.md` - Visual workflow & configuration profiles
- `COMMANDS_SUMMARY.md` - Complete command reference
- `RUN_EXPERIMENTS.md` - Detailed explanations

---

## 🎯 START NOW

Just copy & paste this:

```bash
cd /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3 && \
source .venv/bin/activate && \
./QUICK_START.sh
```

Then grab coffee ☕ and come back in ~2 hours!

