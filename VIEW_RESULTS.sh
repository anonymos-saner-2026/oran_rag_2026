#!/bin/bash
# Helper script to view experiment results
# Usage: ./VIEW_RESULTS.sh [results_dir]

if [ -z "$1" ]; then
    # Find most recent evaluation
    RESULTS_DIR=$(find data/reports -maxdepth 1 -type d -name "eval_*" -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -z "$RESULTS_DIR" ]; then
        echo "No evaluation results found in data/reports/"
        exit 1
    fi
else
    RESULTS_DIR="$1"
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Directory not found: $RESULTS_DIR"
    exit 1
fi

echo "=================================================="
echo "O-RAN RAG Evaluation Results"
echo "=================================================="
echo "Directory: $RESULTS_DIR"
echo ""

# Check if multi-GPU or single results
if [ -f "$RESULTS_DIR/aggregate.summary.json" ]; then
    echo "=== OVERALL METRICS (Multi-GPU) ==="
    jq . "$RESULTS_DIR/aggregate.summary.json"
    echo ""
    
    echo "=== PER-SPLIT BREAKDOWN ==="
    jq '.results[] | {path: .path, questions: .n, accuracy: (.acc | round/1000), avg_citations: .avg_citations}' "$RESULTS_DIR/aggregate.summary.json"
elif [ -f "$RESULTS_DIR/single.summary.json" ]; then
    echo "=== OVERALL METRICS (Single-GPU) ==="
    jq . "$RESULTS_DIR/single.summary.json"
fi

echo ""
echo "=== PER-QUESTION SAMPLES ==="
JSONL_FILE=$(ls "$RESULTS_DIR"/*.jsonl | head -1)
if [ -f "$JSONL_FILE" ]; then
    echo "Showing first 5 questions:"
    jq -c '{i: .global_i, question: .question, gold: .gold, pred: .pred_num, acc: .acc, cites: .citations_count}' "$JSONL_FILE" | head -5
    echo ""
    
    echo "=== ACCURACY BREAKDOWN ==="
    TOTAL=$(jq -c '.' "$RESULTS_DIR"/*.jsonl | wc -l)
    CORRECT=$(jq -c 'select(.acc==1)' "$RESULTS_DIR"/*.jsonl | wc -l)
    echo "Correct: $CORRECT / $TOTAL = $(echo "scale=2; $CORRECT * 100 / $TOTAL" | bc)%"
    echo ""
    
    echo "=== CITATION STATISTICS ==="
    jq -r '.citations_count' "$RESULTS_DIR"/*.jsonl | awk '{sum+=$1; count++} END {print "Average citations: " sum/count}'
    echo ""
    
    echo "=== FAILED PREDICTIONS (acc=0) ==="
    FAILED_COUNT=$(jq -c 'select(.acc==0)' "$RESULTS_DIR"/*.jsonl | wc -l)
    echo "Failed: $FAILED_COUNT questions"
    if [ "$FAILED_COUNT" -gt 0 ]; then
        echo "Sample failures:"
        jq -c 'select(.acc==0) | {q: .question, gold: .gold, pred: .pred_num}' "$RESULTS_DIR"/*.jsonl | head -3
    fi
    echo ""
    
    echo "=== RETRIEVAL DEBUG (TOP PREDICTIONS) ==="
    echo "Sample retrieval rankings (fused vs reranked):"
    jq -c '.[] | select(.acc==1) | ._debug | {fused_top: .fused_top[0:2], reranked_top: .reranked_top[0:2]}' "$RESULTS_DIR"/*.jsonl | head -3
fi

echo ""
echo "=== LOG FILES ==="
ls -lh "$RESULTS_DIR"/*.log.txt 2>/dev/null | awk '{print $9, "(" $5 ")"}'

echo ""
echo "=== QUICK COMMANDS ==="
echo "View all metrics:"
echo "  jq . $RESULTS_DIR/aggregate.summary.json"
echo ""
echo "View specific predictions:"
echo "  jq -c '{q: .question, gold: .gold, pred: .pred_num, acc: .acc}' $RESULTS_DIR/*.jsonl | head -20"
echo ""
echo "View failed answers:"
echo "  jq -c 'select(.acc==0)' $RESULTS_DIR/*.jsonl"
echo ""
echo "View with citations:"
echo "  jq -c '{q: .question, pred: .pred_num, cites: .citations_count, refs: .citations[0:2]}' $RESULTS_DIR/*.jsonl | head -10"
echo ""
