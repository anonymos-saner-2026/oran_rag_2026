set -euo pipefail
PYTHONPATH=src python -m oran_rag.ingest.build_corpus --config configs/app.yaml