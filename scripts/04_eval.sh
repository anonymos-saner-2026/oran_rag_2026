#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m oran_rag.eval.run_eval --config configs/app.yaml --eval data/eval/oran_eval.jsonl
