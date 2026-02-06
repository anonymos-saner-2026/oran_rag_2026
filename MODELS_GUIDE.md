# Running Experiments with Different Models

Now you can easily run experiments with different models! Here's how:

## 🚀 Quick Examples

### Run with Qwen3-4B (Default)
```bash
./QUICK_START.sh
```

### Run with Mistral-7B
```bash
./QUICK_START.sh --model mistralai/Mistral-7B-v0.1
```

### Run with Google Gemma-2-9B
```bash
./QUICK_START.sh --model google/gemma-2-9b
```

### Run with Custom Model
```bash
./QUICK_START.sh --model meta-llama/Llama-2-7b-chat-hf
```

### Specify Custom GPU Set
```bash
./QUICK_START.sh --model mistral/7b --gpus 0,1,2,3,4,5,6,7
```

### Using Environment Variable for GPUs
```bash
GPUS=0,1,2,3,4,5,6,7 ./QUICK_START.sh --model mistral/7b
```

---

## 🔄 Run Multiple Models in Parallel

### Run 2 Models (Qwen3-4B + Mistral-7B) in Parallel on 8 GPUs

```bash
# Method 1: Default models (Qwen3-4B on GPU 0-3, Mistral-7B on GPU 4-7)
./PARALLEL_MODELS.sh

# Method 2: Specify custom models
./PARALLEL_MODELS.sh --models "Qwen/Qwen3-4B-Instruct-2507" "mistral/7b"

# Method 3: With custom GPU allocation
GPUS_TOTAL=8 GPUS_PER_MODEL=4 ./PARALLEL_MODELS.sh
```

### Run 4 Models in Parallel (2 GPUs each on 8 GPUs total)
```bash
GPUS_PER_MODEL=2 ./PARALLEL_MODELS.sh \
  --models "Qwen/Qwen3-4B-Instruct-2507" \
           "mistral/7b" \
           "google/gemma-2-9b" \
           "meta-llama/Llama-2-7b-chat-hf"
```

### Run Single Model on 8 GPUs (faster)
```bash
GPUS_TOTAL=8 GPUS_PER_MODEL=8 ./QUICK_START.sh --model mistral/7b
```

---

## 📊 Compare Results Across Models

After running multiple models:

```bash
# View all results
for model_dir in data/reports/eval_oran_*; do
    echo "=== $(basename $model_dir) ==="
    jq '{total_n, micro_acc, micro_avg_citations}' "$model_dir/aggregate.summary.json"
done

# Compare accuracy across models
echo "Model | Accuracy | Citations"
echo "------|----------|----------"
for model_dir in data/reports/eval_oran_*; do
    model=$(basename "$model_dir" | sed 's/eval_oran_//;s/_[0-9].*$//')
    acc=$(jq '.micro_acc' "$model_dir/aggregate.summary.json")
    cites=$(jq '.micro_avg_citations' "$model_dir/aggregate.summary.json")
    printf "%s | %.2f | %.2f\n" "$model" "$acc" "$cites"
done
```

---

## 🎯 Common Use Cases

### Benchmark 3 Models (4 GPUs each)
You have 12 GPUs? Run all 3 models at once:
```bash
GPUS_TOTAL=12 GPUS_PER_MODEL=4 ./PARALLEL_MODELS.sh \
  --models "Qwen/Qwen3-4B-Instruct-2507" \
           "mistral/7b" \
           "google/gemma-2-9b"
```

### Quick Test Different Models (1 GPU each)
```bash
# Qwen3-4B
GPUS=0 ./QUICK_START.sh --model Qwen/Qwen3-4B-Instruct-2507 &

# Mistral-7B
GPUS=1 ./QUICK_START.sh --model mistral/7b &

# Gemma-2-9B
GPUS=2 ./QUICK_START.sh --model google/gemma-2-9b &

wait
```

### Speed Test (Large Model)
```bash
# Run Mistral-7B on all 8 GPUs for speed
GPUS=0,1,2,3,4,5,6,7 ./QUICK_START.sh --model mistral/7b
```

### Quality vs Speed Tradeoff
```bash
# Small model (fast): Qwen3-4B on 4 GPUs
./QUICK_START.sh --model Qwen/Qwen3-4B-Instruct-2507 --gpus 0,1,2,3

# Large model (slower but better): Mistral-7B on 2 GPUs
./QUICK_START.sh --model mistral/7b --gpus 4,5
```

---

## 📝 Supported Models

Any HuggingFace model ID works! Examples:

### Small Models (Fast)
- `Qwen/Qwen3-4B-Instruct-2507`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `microsoft/phi-2`

### Medium Models (Balanced)
- `mistral/7b` or `mistralai/Mistral-7B-Instruct-v0.1`
- `google/gemma-2-9b`
- `meta-llama/Llama-2-7b-chat-hf`

### Large Models (Slower but Better Quality)
- `mistral/mixtral` or `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `meta-llama/Llama-2-13b-chat-hf`
- `tiiuae/falcon-40b`

---

## ⚡ GPU Allocation Strategies

### Strategy 1: Maximum Throughput (Multiple Models)
```bash
# Run 2 models, 4 GPUs each (8 total)
./PARALLEL_MODELS.sh
```

### Strategy 2: Maximum Speed (Single Model)
```bash
# Run 1 model on all 8 GPUs
GPUS=0,1,2,3,4,5,6,7 ./QUICK_START.sh --model mistral/7b
```

### Strategy 3: Mixed Workloads
```bash
# Large model on 6 GPUs (slower but better quality)
GPUS=0,1,2,3,4,5 ./QUICK_START.sh --model mistral/7b &

# Small model on 2 GPUs (faster)
GPUS=6,7 ./QUICK_START.sh --model Qwen/Qwen3-4B-Instruct-2507 &

wait
```

---

## 📊 Expected Results by Model

| Model | Accuracy | Speed | Memory | Best For |
|-------|----------|-------|--------|----------|
| Qwen3-4B | 65-75% | ⚡⚡⚡ | Low | Quick tests |
| Mistral-7B | 70-78% | ⚡⚡ | Medium | Balanced |
| Gemma-2-9B | 72-80% | ⚡ | Medium | Quality |
| Llama-2-7B | 68-76% | ⚡⚡ | Medium | Balanced |
| Mixtral-8x7B | 75-82% | Slow | High | Best quality |

---

## 🔧 Advanced Options

### Use Custom Config Per Model

Create `configs/app_mistral.yaml`:
```yaml
model:
  backend: local
  name: mistral/7b
  max_new_tokens: 768      # Larger output for 7B
  temperature: 0.25

retrieval:
  enable_dense: true
  enable_rerank: true
```

Then run with:
```bash
PYTHONPATH=src python -m oran_rag.eval.run_eval \
  --config configs/app_mistral.yaml \
  --bench_dir data/benchmarks/oranbench \
  --splits E,M,H \
  --gpus 0,1,2,3
```

### Monitor Multiple Runs
```bash
# In one terminal, watch GPU usage
watch -n 1 'nvidia-smi'

# In another terminal, start parallel experiments
./PARALLEL_MODELS.sh
```

---

## 🐛 Troubleshooting

### Model Download Timeout
```bash
# Pre-download the model before running
python -c "from transformers import AutoModelForCausalLM; \
AutoModelForCausalLM.from_pretrained('mistral/7b')"

# Then run experiments
./QUICK_START.sh --model mistral/7b
```

### Parallel Job Fails
```bash
# Check which model failed
grep -r "ERROR" data/reports/*/shard_*_gpu*.log.txt

# Run that model individually for debugging
./QUICK_START.sh --model <failing_model>
```

### GPU Memory Issues
```bash
# Reduce max_new_tokens in config or use smaller model
./QUICK_START.sh --model Qwen/Qwen3-4B-Instruct-2507

# Or reduce context
# Edit configs/app.yaml: packing.max_context_chars: 15000
```

---

## 📈 Creating a Model Comparison Report

```bash
#!/bin/bash
# Save as compare_models.sh

echo "Model Comparison Report"
echo "======================"
echo ""

for model_dir in data/reports/eval_oran_*; do
    name=$(basename "$model_dir" | sed 's/eval_oran_//;s/_[0-9].*$//')
    summary="$model_dir/aggregate.summary.json"
    
    if [ -f "$summary" ]; then
        echo "## $name"
        total=$(jq '.total_n' "$summary")
        acc=$(jq '.micro_acc' "$summary")
        cites=$(jq '.micro_avg_citations' "$summary")
        
        echo "- Questions: $total"
        echo "- Accuracy: $(echo "$acc * 100" | bc)%"
        echo "- Avg Citations: $cites"
        echo ""
        
        echo "### Per-Split Performance"
        jq -r '.results[] | "- \(.path): \(.acc * 100 | round)% accuracy (\(.avg_citations) citations)"' "$summary"
        echo ""
    fi
done
```

Run it:
```bash
bash compare_models.sh
```

---

## 🎯 Examples

### Example 1: Quick Comparison (4 hours)
Run 2 models, 4 GPUs each, on 8 GPUs total:
```bash
./PARALLEL_MODELS.sh
```

### Example 2: Comprehensive Benchmark (6 hours)
Run 3 models, 2 GPUs each on 6 total GPUs:
```bash
GPUS_PER_MODEL=2 ./PARALLEL_MODELS.sh \
  --models "Qwen/Qwen3-4B-Instruct-2507" \
           "mistral/7b" \
           "google/gemma-2-9b"
```

### Example 3: Speed Test (1 hour)
Run 1 large model on all GPUs:
```bash
GPUS=0,1,2,3,4,5,6,7 ./QUICK_START.sh --model mistral/7b
```

### Example 4: Individual Tests (Sequential)
```bash
# Run each model sequentially
for model in "Qwen/Qwen3-4B-Instruct-2507" "mistral/7b" "google/gemma-2-9b"; do
    echo "Running $model..."
    ./QUICK_START.sh --model "$model" --gpus 0,1,2,3
    echo "Done with $model"
done
```

---

## 💡 Pro Tips

1. **Share corpus & indexes**: The first model builds them. Subsequent models reuse them automatically!

2. **Monitor with tmux**: Run each model in a separate tmux session
   ```bash
   tmux new-session -d -s qwen "./QUICK_START.sh --model Qwen/Qwen3-4B-Instruct-2507"
   tmux new-session -d -s mistral "./QUICK_START.sh --model mistral/7b"
   tmux list-sessions
   ```

3. **Compare side-by-side**:
   ```bash
   paste <(jq -r '.results[] | "\(.path): \(.acc)"' data/reports/eval_oran_Qwen*) \
         <(jq -r '.results[] | "\(.path): \(.acc)"' data/reports/eval_oran_mistral*)
   ```

4. **Save results**:
   ```bash
   mkdir -p results_archive
   cp -r data/reports/eval_oran_* results_archive/
   ```

