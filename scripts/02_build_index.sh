#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m oran_rag.index.bm25_index build --config configs/app.yaml
PYTHONPATH=src python -m oran_rag.index.dense_index build --config configs/app.yaml
