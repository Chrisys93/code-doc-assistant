#!/usr/bin/env bash
# ============================================================
# run.sh — Launch Code Documentation Assistant
# ============================================================
# Auto-detects GPU availability and selects the appropriate
# Docker Compose configuration.
#
# Usage:
#   ./run.sh                    # auto-detect GPU, lightweight tier
#   ./run.sh --gpu              # force GPU mode
#   MODEL_TIER=full ./run.sh    # full tier (needs GPU)
# ============================================================

set -e

export MODEL_TIER="${MODEL_TIER:-lightweight}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"

# Detect GPU
GPU_AVAILABLE=false
COMPOSE_FILES="-f docker-compose.yml"

if [ "$1" = "--gpu" ]; then
    GPU_AVAILABLE=true
elif command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_AVAILABLE=true
fi

if [ "$GPU_AVAILABLE" = true ]; then
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.gpu.yml"
    echo "=== Code Documentation Assistant (GPU mode) ==="
else
    echo "=== Code Documentation Assistant (CPU mode) ==="
    if [ "$MODEL_TIER" = "full" ]; then
        echo "WARNING: Full tier on CPU will be very slow. Consider MODEL_TIER=lightweight"
    fi
fi

echo "  Model tier:      ${MODEL_TIER}"
echo "  Embedding model: ${EMBEDDING_MODEL}"
echo "  GPU:             ${GPU_AVAILABLE}"
echo "  UI:              http://localhost:8501"
echo ""
echo "First run will pull models — this may take a few minutes."
echo "================================================"
echo ""

docker compose ${COMPOSE_FILES} up --build
