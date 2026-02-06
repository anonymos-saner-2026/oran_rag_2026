#!/bin/bash
# Run O-RAN RAG Experiments in Parallel with Multiple Models
#
# This script runs experiments with different models in parallel
# Usage:
#   ./PARALLEL_MODELS.sh
#   GPUS_PER_MODEL=2 ./PARALLEL_MODELS.sh    # 2 GPUs per model
#   ./PARALLEL_MODELS.sh --models "Qwen/Qwen3-4B-Instruct-2507" "mistral/7b"

set -euo pipefail

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS_TOTAL="${GPUS_TOTAL:-8}"  # Total GPUs available (default 8)
GPUS_PER_MODEL="${GPUS_PER_MODEL:-4}"  # GPUs per model (default 4)

# Parse models from arguments or use defaults
MODELS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --gpus-total)
            GPUS_TOTAL="$2"
            shift 2
            ;;
        --gpus-per-model)
            GPUS_PER_MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default models if not specified
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(
        "Qwen/Qwen3-4B-Instruct-2507"
        "mistral/7b"
    )
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Running Parallel O-RAN RAG Experiments with Multiple Models║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Total GPUs available: $GPUS_TOTAL"
echo "GPUs per model: $GPUS_PER_MODEL"
echo "Models to test: ${MODELS[@]}"
echo ""

# Validate GPU configuration
NUM_MODELS=${#MODELS[@]}
REQUIRED_GPUS=$((NUM_MODELS * GPUS_PER_MODEL))
if [ "$REQUIRED_GPUS" -gt "$GPUS_TOTAL" ]; then
    echo "ERROR: Need $REQUIRED_GPUS GPUs ($NUM_MODELS models × $GPUS_PER_MODEL GPUs), but only have $GPUS_TOTAL"
    echo "Options:"
    echo "  - Reduce number of models"
    echo "  - Reduce GPUS_PER_MODEL: GPUS_PER_MODEL=2 ./PARALLEL_MODELS.sh"
    echo "  - Increase GPUS_TOTAL: GPUS_TOTAL=16 ./PARALLEL_MODELS.sh"
    exit 1
fi

echo "✓ Configuration valid: $NUM_MODELS models × $GPUS_PER_MODEL GPUs = $REQUIRED_GPUS GPUs (available: $GPUS_TOTAL)"
echo ""

# Prepare data once (shared by all models)
echo "════════════════════════════════════════════════════════════════"
echo "STEP 1: Preparing Data (shared, one-time)"
echo "════════════════════════════════════════════════════════════════"
source .venv/bin/activate 2>/dev/null || true
mkdir -p data/raw_specs
if [ ! -f "data/raw_specs/$(ls data/full_oran_specs/*.pdf 2>/dev/null | head -1 | xargs basename)" ]; then
    echo "Copying O-RAN specs to data/raw_specs/..."
    cp data/full_oran_specs/*.pdf data/raw_specs/ 2>/dev/null || echo "Specs already present"
fi

# Build corpus and indexes once (shared by all models)
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "STEP 2: Building Corpus & Indexes (shared, one-time)"
echo "════════════════════════════════════════════════════════════════"
if [ ! -f "data/intermediate/chunks.jsonl" ]; then
    echo "Building corpus..."
    PYTHONPATH=src python -m oran_rag.ingest.build_corpus --config configs/app.yaml
fi

if [ ! -f "data/indexes/bm25/bm25.pkl" ] || [ ! -f "data/indexes/faiss/faiss.index" ]; then
    echo "Building indexes..."
    PYTHONPATH=src python -m oran_rag.index.build --config configs/app.yaml
fi

# Run evaluations in parallel
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "STEP 3: Running Evaluations in Parallel"
echo "════════════════════════════════════════════════════════════════"

PIDS=()
GPU_START=0

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    GPU_END=$((GPU_START + GPUS_PER_MODEL - 1))
    GPUS_FOR_MODEL=$(seq -s, $GPU_START $GPU_END)
    
    echo ""
    echo "Starting evaluation for: $MODEL"
    echo "  Using GPUs: $GPUS_FOR_MODEL"
    
    # Run in background
    (
        export GPUS="$GPUS_FOR_MODEL"
        cd "$PROJECT_DIR"
        ./QUICK_START.sh --model "$MODEL"
    ) &
    
    PIDS+=($!)
    GPU_START=$((GPU_END + 1))
done

echo ""
echo "All experiments started. PIDs: ${PIDS[@]}"
echo "Waiting for all to complete..."
echo ""

# Wait for all background jobs
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    MODEL="${MODELS[$i]}"
    
    if wait "$PID"; then
        echo "✓ [$((i+1))/${#MODELS[@]}] $MODEL - COMPLETED"
    else
        echo "✗ [$((i+1))/${#MODELS[@]}] $MODEL - FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "STEP 4: Summary of Results"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Results Summary:"
    for MODEL in "${MODELS[@]}"; do
        RESULT_DIR=$(find data/reports -maxdepth 1 -type d -name "eval_oran_${MODEL//\//_}_*" -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
        if [ -f "$RESULT_DIR/aggregate.summary.json" ]; then
            echo ""
            echo "Model: $MODEL"
            jq '{total_n, micro_acc, micro_avg_citations}' "$RESULT_DIR/aggregate.summary.json"
        fi
    done
else
    echo "✗ $FAILED EXPERIMENTS FAILED"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Detailed comparison command:"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "for model in ${MODELS[@]}; do"
echo "  echo \"Model: \$model\""
echo "  jq . data/reports/eval_oran_\${model//\//_}_*/*.summary.json | head -20"
echo "done"
echo ""
