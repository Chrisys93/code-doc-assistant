#!/usr/bin/env bash
# ============================================================
# run-dev.sh — Launch the full DEV environment
# ============================================================
# Layers the dev overlay (MLflow + src live-reload) on top of the
# base stack, auto-detecting GPU exactly like run.sh:
#
#   base  ->  docker-compose.yml
#   +gpu  ->  docker-compose.gpu.yml      (if an NVIDIA GPU is present)
#   +dev  ->  docker-compose.dev.yml      (MLflow, live-reload)
#
# Usage:
#   ./run-dev.sh                     # auto-detect GPU
#   ./run-dev.sh --gpu               # force GPU mode
#   ./run-dev.sh -d                  # detached (background)
#   MODEL_TIER=lightweight ./run-dev.sh
#
# Access:
#   Streamlit UI:  http://localhost:8501
#   MLflow UI:     http://localhost:5000
# ============================================================

set -e

export EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"

# Parse flags
FORCE_GPU=false
DETACH=""
for arg in "$@"; do
    case "$arg" in
        --gpu) FORCE_GPU=true ;;
        -d|--detach) DETACH="-d" ;;
    esac
done

# Detect GPU
GPU_AVAILABLE=false
if [ "$FORCE_GPU" = true ]; then
    GPU_AVAILABLE=true
elif command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_AVAILABLE=true
fi

# Default tier follows hardware: full on GPU, lightweight on CPU (overridable).
if [ "$GPU_AVAILABLE" = true ]; then
    export MODEL_TIER="${MODEL_TIER:-full}"
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.dev.yml"
    MODE="GPU + dev"
else
    export MODEL_TIER="${MODEL_TIER:-lightweight}"
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.dev.yml"
    MODE="CPU + dev"
    if [ "$MODEL_TIER" = "full" ]; then
        echo "WARNING: Full tier on CPU will be very slow. Consider MODEL_TIER=lightweight"
    fi
fi

echo "=== Code Documentation Assistant (${MODE}) ==="
echo "  Model tier:      ${MODEL_TIER}"
echo "  Embedding model: ${EMBEDDING_MODEL}"
echo "  GPU:             ${GPU_AVAILABLE}"
echo "  Streamlit UI:    http://localhost:8501"
echo "  MLflow UI:       http://localhost:5000"
echo ""
echo "First run will pull models — this may take a few minutes."
echo "================================================"
echo ""

docker compose ${COMPOSE_FILES} up --build ${DETACH}
