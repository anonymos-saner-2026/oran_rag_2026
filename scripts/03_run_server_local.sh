#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m oran_rag.api.server --config configs/app.yaml
