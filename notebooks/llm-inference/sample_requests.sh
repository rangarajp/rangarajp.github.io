#!/bin/bash
set -e

BASE_URL="http://127.0.0.1:8000"

if [ -z "${GATEWAY_API_KEY:-}" ]; then
  echo "GATEWAY_API_KEY must be set before running this script." >&2
  exit 1
fi

AUTH_HEADER=(-H "X-API-Key: $GATEWAY_API_KEY")

printf '\nHealth check:\n'
curl --noproxy '*' "$BASE_URL/health"
echo

printf '\nModels:\n'
curl --noproxy '*' "$BASE_URL/models"
echo

printf '\nStatus:\n'
curl --noproxy '*' "$BASE_URL/status"
echo

printf '\nQwen explicit routing:\n'
curl --noproxy '*' "${AUTH_HEADER[@]}" "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "prompt": "Explain the KV cache in one sentence.",
    "max_tokens": 20,
    "temperature": 0.0
  }'
echo

printf '\nTinyLlama explicit routing:\n'
curl --noproxy '*' "${AUTH_HEADER[@]}" "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "prompt": "Explain batching in one sentence.",
    "max_tokens": 20,
    "temperature": 0.0
  }'
echo

printf '\nAutomatic routing with short prompt:\n'
curl --noproxy '*' "${AUTH_HEADER[@]}" "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is KV cache?",
    "max_tokens": 20,
    "temperature": 0.0
  }'
echo

printf '\nAutomatic routing with long prompt:\n'
curl --noproxy '*' "${AUTH_HEADER[@]}" "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain transformer inference in detail. Explain transformer inference in detail. Explain transformer inference in detail. Explain transformer inference in detail. Explain transformer inference in detail. Explain transformer inference in detail.",
    "max_tokens": 20,
    "temperature": 0.0
  }'
echo

printf '\nMetrics:\n'
curl --noproxy '*' "$BASE_URL/metrics"
echo
