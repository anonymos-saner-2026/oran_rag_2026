#!/bin/bash
# Quick Start: Run Full O-RAN RAG Experiments
# This script automates the complete pipeline from data preparation to metric reporting
# 
# Usage:
#   ./QUICK_START.sh                              # Default: Qwen3-4B on 4 GPUs
#   ./QUICK_START.sh --model mistral/7b          # Mistral-7B on 4 GPUs
#   ./QUICK_START.sh --model google/gemma-2-9b   # Gemma-2-9B on 4 GPUs
#   GPUS=0,1,2,3,4,5,6,7 ./QUICK_START.sh --model mistral/7b  # On 8 GPUs

set -euo pipefail

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL_SPECS_DIR="/home/tungnvt5re/Downloads/kiwi_momo/oran-rag-qwen3/cm/archive/tungnvt5_node3/oran_rag/oran-rag-qwen3/data/full_oran_specs"
GPUS="${GPUS:-0,1,2,3}"  # Default to 4 GPUs
MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"  # Default model
CONFIG_FILE="configs/app.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./QUICK_START.sh [--model MODEL_NAME] [--gpus GPU_LIST]"
            echo "Examples:"
            echo "  ./QUICK_START.sh"
            echo "  ./QUICK_START.sh --model mistral/7b"
            echo "  ./QUICK_START.sh --model google/gemma-2-9b --gpus 0,1,2,3,4,5,6,7"
            exit 1
            ;;
    esac
done

# Create temporary config with the specified model
TEMP_CONFIG="/tmp/app_${MODEL//\//_}_${TIMESTAMP}.yaml"
cp "$CONFIG_FILE" "$TEMP_CONFIG"
sed -i "s|name: .*|name: $MODEL|" "$TEMP_CONFIG"
CONFIG_FILE="$TEMP_CONFIG"

echo "=================================================="
echo "O-RAN RAG Experiments - $MODEL"
echo "=================================================="
echo "Project: $PROJECT_DIR"
echo "Model: $MODEL"
echo "Specs: $FULL_SPECS_DIR"
echo "GPUs: $GPUS"
echo "Config: $CONFIG_FILE"
echo "Timestamp: $TIMESTAMP"
echo "=================================================="

cd "$PROJECT_DIR"

# Step 1: Activate environment
echo ""
echo "[STEP 1/5] Setting up Python environment..."
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv .venv
fi
source .venv/bin/activate

# Step 2: Prepare data
echo ""
echo "[STEP 2/5] Preparing data - copying full O-RAN specs..."
mkdir -p data/raw_specs
cp -v "$FULL_SPECS_DIR"/*.pdf data/raw_specs/ 2>/dev/null || echo "Warning: Some files may not be PDF"
echo "Copied $(ls data/raw_specs/*.pdf 2>/dev/null | wc -l) PDF files"

# Step 3: Build corpus
echo ""
echo "[STEP 3/5] Building corpus (extracting & chunking PDFs)..."
echo "This may take 10-30 minutes..."
PYTHONPATH=src python -m oran_rag.ingest.build_corpus --config "$CONFIG_FILE" || {
    echo "ERROR: Corpus building failed"
    rm -f "$TEMP_CONFIG" 2>/dev/null || true
    exit 1
}
echo "✓ Corpus built successfully"
echo "  Chunks: $(wc -l < data/intermediate/chunks.jsonl) chunks"

# Step 4: Build indexes
echo ""
echo "[STEP 4/5] Building indexes (BM25 + FAISS)..."
echo "This may take 5-20 minutes..."
PYTHONPATH=src python -m oran_rag.index.build --config "$CONFIG_FILE" || {
    echo "ERROR: Index building failed"
    rm -f "$TEMP_CONFIG" 2>/dev/null || true
    exit 1
}
echo "✓ Indexes built successfully"

# Step 5: Run evaluation
echo ""
echo "[STEP 5/5] Running evaluation on GPUs: $GPUS"
OUT_DIR="data/reports/eval_oran_${MODEL//\//_}_$TIMESTAMP"
mkdir -p "$OUT_DIR"

PYTHONPATH=src python -m oran_rag.eval.run_eval \
    --config "$CONFIG_FILE" \
    --bench_dir data/benchmarks/oranbench \
    --splits E,M,H \
    --gpus "$GPUS" \
    --top_k 10 \
    --out_dir "$OUT_DIR" \
    --log_raw 1 \
    --log_prompt 0 || {
    echo "ERROR: Evaluation failed"
    rm -f "$TEMP_CONFIG" 2>/dev/null || true
    exit 1
}

# Final Summary
echo ""
echo "=================================================="
echo "✓ EXPERIMENTS COMPLETE!"
echo "=================================================="
echo ""
echo "Results Directory: $OUT_DIR"
echo "Model: $MODEL"
echo ""
echo "METRICS SUMMARY:"
jq . "$OUT_DIR/aggregate.summary.json"
echo ""
echo "QUICK ANALYSIS:"
echo "  - Total questions: $(jq '.total_n' "$OUT_DIR/aggregate.summary.json")"
echo "  - Overall accuracy: $(jq '.micro_acc' "$OUT_DIR/aggregate.summary.json")"
echo "  - Avg citations per answer: $(jq '.micro_avg_citations' "$OUT_DIR/aggregate.summary.json")"
echo ""
echo "VIEW RESULTS:"
echo "  Cat metrics: cat $OUT_DIR/aggregate.summary.json | jq ."
echo "  View per-question: jq -c '{question, gold, pred_num, acc}' $OUT_DIR/*.jsonl | head -20"
echo "  View retrieval debug: jq -c '{question, _debug}' $OUT_DIR/*.jsonl | head -5"
echo ""
echo "LOG FILES:"
echo "  - GPU logs: $OUT_DIR/shard_*_gpu*.log.txt"
echo "  - Detailed results: $OUT_DIR/shard_*_gpu*.jsonl"
echo "  - Subprocess logs: $OUT_DIR/shard_*_gpu*.subprocess.log.txt"
echo ""
echo "=================================================="

# Cleanup temp config
rm -f "$TEMP_CONFIG" 2>/dev/null || true
