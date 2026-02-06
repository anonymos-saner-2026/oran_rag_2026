# 📋 Complete Summary: All Commands to Run Experiments

> Last updated: February 4, 2026
> For: Qwen3-4B on Full O-RAN Specifications

---

## 🎯 TL;DR - Just Copy This

```bash
cd /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3
source .venv/bin/activate 2>/dev/null || (python3.10 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -r requirements.txt)
./QUICK_START.sh
```

That's it. Everything else is optional details below.

---

## 📚 Documentation Files Created

| File | Purpose | Read When |
|------|---------|-----------|
| [START_HERE.md](START_HERE.md) | **Quick start guide** | First time |
| [QUICK_START.sh](QUICK_START.sh) | **Automated pipeline** | Want everything in one command |
| [VIEW_RESULTS.sh](VIEW_RESULTS.sh) | **Result analysis** | After experiments finish |
| [WORKFLOW.md](WORKFLOW.md) | Visual workflow & profiles | Want to understand the pipeline |
| [COMMANDS_SUMMARY.md](COMMANDS_SUMMARY.md) | Complete command reference | Need all commands |
| [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md) | Detailed guide | Deep dive |

---

## 🚀 The Complete Command Sequence

### Prerequisites (One-time Setup)

```bash
cd /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3

# Create Python environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

### Main Pipeline (Run This Each Time)

#### Option 1: Automatic (Recommended)
```bash
source .venv/bin/activate
./QUICK_START.sh
```

**What it does:**
1. Copies PDFs from `data/full_oran_specs/` to `data/raw_specs/`
2. Builds corpus (extracts & chunks PDFs)
3. Builds indexes (BM25 + FAISS)
4. Runs evaluation on 4 GPUs
5. Shows results

**Expected time:** 2-3 hours

---

#### Option 2: Manual Steps (More Control)

**Step 1: Prepare Data**
```bash
source .venv/bin/activate
mkdir -p data/raw_specs
cp data/full_oran_specs/*.pdf data/raw_specs/
echo "Copied $(ls data/raw_specs/*.pdf | wc -l) PDFs"
```

**Step 2: Build Corpus (20-30 min)**
```bash
bash scripts/01_build_corpus.sh
```

**Step 3: Build Indexes (10-20 min)**
```bash
bash scripts/02_build_index.sh
```

**Step 4: Run Evaluation**

**For 4 GPUs:**
```bash
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --gpus 0,1,2,3 \
  --top_k 10 \
  --out_dir data/reports/eval_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 \
  --log_prompt 0
```

**For 8 GPUs (Faster):**
```bash
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --gpus 0,1,2,3,4,5,6,7 \
  --top_k 10 \
  --out_dir data/reports/eval_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 \
  --log_prompt 0
```

**For Single GPU (Testing):**
```bash
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app.yaml \
  --eval data/eval/oran_eval.jsonl \
  --top_k 10 \
  --out_dir data/reports/eval_test_$(date +%Y%m%d_%H%M%S) \
  --log_raw 1 \
  --log_prompt 0
```

---

#### Option 3: Quick Test First

```bash
source .venv/bin/activate
python scripts/mini_run_eval.py \
  --config configs/app.yaml \
  --eval data/eval/oran_eval.jsonl \
  --top_k 10 \
  --out_dir results/test_$(date +%Y%m%d_%H%M%S)

cat results/test_*/mini_eval_*.summary.json | jq .
```

---

## 📊 View Results

### Automatic (Recommended)
```bash
./VIEW_RESULTS.sh
```

### Manual Analysis

**View Overall Metrics:**
```bash
jq . data/reports/eval_mgpu_*/aggregate.summary.json
```

**View Accuracy by Difficulty:**
```bash
jq '.results[] | {split: .path, accuracy: (.acc*100|round), citations: .avg_citations}' \
  data/reports/eval_mgpu_*/aggregate.summary.json
```

**View Individual Questions:**
```bash
jq -c '{question, gold, predicted: .pred_num, correct: .acc}' \
  data/reports/eval_mgpu_*/*.jsonl | head -30
```

**View Failed Predictions:**
```bash
jq -c 'select(.acc==0) | {q: .question, gold, pred: .pred_num}' \
  data/reports/eval_mgpu_*/*.jsonl
```

**View Citations:**
```bash
jq -c '{q: .question, citations: .citations_count, refs: (.citations[0:2] | map(.chunk_id))}' \
  data/reports/eval_mgpu_*/*.jsonl | head -10
```

---

## 🔧 Configuration Tuning

Edit `configs/app.yaml` for different experiments:

### For Speed (Lower Quality)
```yaml
model:
  max_new_tokens: 256        # reduced from 512
  temperature: 0.1           # more deterministic

retrieval:
  enable_dense: false        # skip dense retrieval
  enable_rerank: false       # skip reranking
  bm25_topk: 30              # reduced from 80

ingest:
  max_pages_per_pdf: 500     # reduced from 2000
```

### For Quality (Slower)
```yaml
model:
  max_new_tokens: 768        # increased from 512
  temperature: 0.3           # more diverse

retrieval:
  enable_dense: true         # keep dense retrieval
  enable_rerank: true        # keep reranking
  bm25_topk: 100             # increased from 80
  dense_topk: 100            # increased from 80

packing:
  max_context_chars: 35000   # increased from 25000
  neighbor_window: 2         # include neighbors
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `max_new_tokens` in config |
| GPU not detected | Run `nvidia-smi` |
| Slow corpus building | Reduce `max_pages_per_pdf` |
| Process stuck | `pkill -f "oran_rag"` |
| Model not found | `python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')"` |

---

## 📂 What Gets Generated

After running experiments:

```
data/reports/eval_mgpu_YYYYMMDD_HHMMSS/
├── aggregate.summary.json          # 📊 MAIN RESULTS
│   {
│     "total_n": 300,
│     "micro_acc": 0.72,             # Overall accuracy
│     "micro_avg_citations": 2.4,
│     "results": [...]
│   }
├── shard_0_gpu0.jsonl              # Per-question results GPU0
├── shard_0_gpu0.summary.json       # GPU0 metrics
├── shard_0_gpu0.log.txt            # GPU0 logs (detailed)
├── shard_1_gpu1.jsonl              # GPU1 results
├── shard_1_gpu1.summary.json       # GPU1 metrics
├── shard_1_gpu1.log.txt            # GPU1 logs
└── ...                             # (one set per GPU)
```

---

## ⏱️ Timeline

| Phase | Time | Note |
|-------|------|------|
| Setup venv + pip | ~10 min | One-time |
| Copy PDFs | ~1 min | Every run |
| Build corpus | 20-30 min | Extract & chunk |
| Build indexes | 10-20 min | BM25 + FAISS |
| Evaluation (4 GPU) | 1-2 hours | Main experiment |
| Evaluation (8 GPU) | 30-60 min | Faster |
| Total | ~2-3 hours | Complete pipeline |

---

## 📈 Expected Results

| Metric | Easy | Medium | Hard | Overall |
|--------|------|--------|------|---------|
| Accuracy | 75-85% | 60-75% | 40-60% | 60-75% |
| Avg Citations | 2.5 | 2.4 | 2.2 | 2.3 |

---

## ✅ Verification Steps

After setup, verify everything works:

```bash
# Check Python environment
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check model can load
python -c "from transformers import AutoModelForCausalLM; print('Qwen3-4B downloadable')"

# Check data is in place
ls data/full_oran_specs/*.pdf | wc -l
```

---

## 🎯 Quick Reference Table

| What | Command |
|------|---------|
| **Everything** | `./QUICK_START.sh` |
| **Corpus only** | `bash scripts/01_build_corpus.sh` |
| **Indexes only** | `bash scripts/02_build_index.sh` |
| **Test only** | `python scripts/mini_run_eval.py --config configs/app.yaml --eval data/eval/oran_eval.jsonl` |
| **Eval (4 GPU)** | `PYTHONPATH=src python -m oran_rag.eval.run_eval --config configs/app.yaml --bench_dir data/benchmarks/oranbench --splits E,M,H --gpus 0,1,2,3` |
| **Eval (8 GPU)** | `PYTHONPATH=src python -m oran_rag.eval.run_eval --config configs/app.yaml --bench_dir data/benchmarks/oranbench --splits E,M,H --gpus 0,1,2,3,4,5,6,7` |
| **View results** | `./VIEW_RESULTS.sh` |
| **View metrics** | `jq . data/reports/eval_mgpu_*/aggregate.summary.json` |

---

## 🔗 Useful Links

- [Full documentation](RUN_EXPERIMENTS.md)
- [Workflow diagram](WORKFLOW.md)
- [All commands](COMMANDS_SUMMARY.md)
- [Quick guide](START_HERE.md)

---

## 💡 Pro Tips

1. **Save results before cleanup:**
   ```bash
   cp -r data/reports/eval_mgpu_20260204_120000 ~/my_results_backup/
   ```

2. **Compare multiple runs:**
   ```bash
   for f in data/reports/eval_*/aggregate.summary.json; do
     echo "=== $f ===" 
     jq '{acc: .micro_acc, cites: .micro_avg_citations}' "$f"
   done
   ```

3. **Monitor during eval:**
   ```bash
   watch -n 1 'nvidia-smi'  # GPU usage
   tail -f data/reports/eval_mgpu_*/shard_0_*.log.txt  # Logs
   ```

4. **Export for analysis:**
   ```bash
   jq -r '.results[] | [.path, .n, .acc] | @csv' \
     data/reports/eval_*/aggregate.summary.json > results.csv
   ```

---

## ✨ Final Notes

- **All PDFs** from `data/full_oran_specs/` will be processed
- **All evaluation splits** (E, M, H) will be evaluated
- **Results** are saved per-GPU and aggregated
- **Configuration** in `configs/app.yaml` controls everything
- **Qwen3-4B** is already configured in the config file

---

## 🚀 Start Now

```bash
cd /home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3
source .venv/bin/activate
./QUICK_START.sh
```

See you in 2-3 hours! ☕

