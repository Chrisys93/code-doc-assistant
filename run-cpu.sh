#!/usr/bin/env bash
# ============================================================
# run-cpu.sh — Launch Code Documentation Assistant (CPU-only)
# ============================================================
# For machines without an NVIDIA GPU.
# Uses the lightweight tier (Phi-3.5 Mini 3.8B) by default.
#
# Requirements: Docker & Docker Compose
# RAM: ~6-8GB used by the full stack
# ============================================================

set -e

export MODEL_TIER="${MODEL_TIER:-lightweight}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"

echo "=== Code Documentation Assistant (CPU mode) ==="
echo "  Model tier:      ${MODEL_TIER}"
echo "  Embedding model: ${EMBEDDING_MODEL}"
echo "  UI:              http://localhost:8501"
echo ""
echo "First run will pull models (~2.5GB) — this may take a few minutes."
echo "================================================"
echo ""

docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
