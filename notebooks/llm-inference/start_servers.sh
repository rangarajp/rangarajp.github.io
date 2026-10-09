#!/bin/bash
set -e

# ============================================================
# Multi-model vLLM server launcher
# ============================================================

# Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_lab

# Prevent ~/.local packages leaking into the environment
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# Offline mode because models are local
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -z "${GATEWAY_API_KEY:-}" ]; then
    echo "GATEWAY_API_KEY must be set before starting the gateway."
    exit 1
fi

# ------------------------------------------------------------
# CONFIGURATION — change these paths
# ------------------------------------------------------------

QWEN_PATH="/home/mbrdiuser/3D_RP/llm_inference/models/Qwen2.5-7B"
TINYLLAMA_PATH="/home/mbrdiuser/3D_RP/llm_inference/models/TinyLlama-1.1B-Chat-v1.0"

LOG_DIR="./logs"
PID_DIR="./pids"

mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

wait_for_endpoint () {
    NAME=$1
    URL=$2
    MAX_ATTEMPTS=60

    echo "Waiting for $NAME to become ready..."

    for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
        if curl --noproxy '*' --silent --show-error --fail "$URL" >/dev/null 2>&1; then
            echo "$NAME is ready."
            return 0
        fi
        sleep 2
    done

    echo "$NAME did not become ready within $((MAX_ATTEMPTS * 2)) seconds."
    return 1
}

wait_for_gateway () {
    MAX_ATTEMPTS=60

    echo "Waiting for gateway and model health checks..."

    for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
        if python -c "import json,sys,urllib.request; opener=urllib.request.build_opener(urllib.request.ProxyHandler({})); data=json.load(opener.open('http://127.0.0.1:8000/health', timeout=3)); sys.exit(0 if data and all(item.get('healthy') for item in data.values()) else 1)" >/dev/null 2>&1; then
            echo "Gateway and model health checks passed."
            return 0
        fi
        sleep 2
    done

    echo "Gateway health checks did not pass within $((MAX_ATTEMPTS * 2)) seconds."
    return 1
}


# ------------------------------------------------------------
# QWEN
# ------------------------------------------------------------

echo "Starting Qwen on port 8001..."

nohup python -m vllm.entrypoints.openai.api_server \
    --model "$QWEN_PATH" \
    --served-model-name qwen2.5-0.5b \
    --host 127.0.0.1 \
    --port 8001 \
    --dtype float16 \
    --max-model-len 512 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.15 \
    --enforce-eager \
    --guided-decoding-backend lm-format-enforcer \
    > "$LOG_DIR/qwen.log" 2>&1 &

echo $! > "$PID_DIR/qwen.pid"

echo "Qwen PID: $(cat "$PID_DIR/qwen.pid")"

wait_for_endpoint "Qwen" "http://127.0.0.1:8001/v1/models"


# ------------------------------------------------------------
# TINYLLAMA
# ------------------------------------------------------------

echo "Starting TinyLlama on port 8002..."

nohup python -m vllm.entrypoints.openai.api_server \
    --model "$TINYLLAMA_PATH" \
    --served-model-name tinyllama-1.1b \
    --host 127.0.0.1 \
    --port 8002 \
    --dtype float16 \
    --max-model-len 512 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.15 \
    --enforce-eager \
    --guided-decoding-backend lm-format-enforcer \
    > "$LOG_DIR/tinyllama.log" 2>&1 &

echo $! > "$PID_DIR/tinyllama.pid"

echo "TinyLlama PID: $(cat "$PID_DIR/tinyllama.pid")"

wait_for_endpoint "TinyLlama" "http://127.0.0.1:8002/v1/models"

echo "Starting gateway on port 8000..."

nohup python -m uvicorn gateway:app \
    --host 127.0.0.1 \
    --port 8000 \
    > "$LOG_DIR/gateway.log" 2>&1 &

echo $! > "$PID_DIR/gateway.pid"
echo "Gateway PID: $(cat "$PID_DIR/gateway.pid")"

wait_for_gateway

echo ""
echo "========================================"
echo "All servers launched"
echo "Gateway   : http://127.0.0.1:8000"
echo "Qwen      : http://127.0.0.1:8001"
echo "TinyLlama : http://127.0.0.1:8002"
echo "========================================"