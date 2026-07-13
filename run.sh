#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Boot llama-server for the Granite Switch agent.
#
# Requires a llama.cpp built from the feature/granite-switch branch (HEAD that
# uses the graniteswitch.adapters.* metadata namespace) and a GGUF converted with
# that same branch's convert_hf_to_gguf.py.
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/Desktop/IBM/projects/llama.cpp}"
SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build/bin/llama-server}"
MODEL="${GS_MODEL:-$(dirname "$0")/granite-switch-4.1-3b.f16.gguf}"
PORT="${GS_PORT:-8080}"

if [[ ! -x "$SERVER" ]]; then
  echo "llama-server not found at $SERVER" >&2
  echo "Build it: cmake --build $LLAMA_DIR/build --target llama-server -j" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Model GGUF not found at $MODEL" >&2
  exit 1
fi

echo "Starting llama-server on :$PORT with $(basename "$MODEL")"
exec "$SERVER" \
  -m "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --jinja \
  -c 8192 \
  -ngl 999
