#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SELFPLAY_ROOT="${SELFPLAY_ROOT:-$HOME/Downloads/selfplay_shards}"
MODEL_PATH="${MODEL_PATH:-$HOME/Downloads/model_v1.pt}"
STOCKFISH_BIN="${STOCKFISH_BIN:-/usr/local/bin/stockfish}"
GAMES_PER_BATCH="${GAMES_PER_BATCH:-200}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-2048}"
STOCKFISH_DEPTH="${STOCKFISH_DEPTH:-10}"
STOCKFISH_MOVETIME="${STOCKFISH_MOVETIME:-0.0}"
POLICY_SOURCE="${POLICY_SOURCE:-stockfish}"
VALUE_SOURCE="${VALUE_SOURCE:-stockfish}"
WORKERS="${WORKERS:-1}"
MCTS_CPUCT="${MCTS_CPUCT:-1.4}"
MCTS_MAX_NODES="${MCTS_MAX_NODES:-96}"
MCTS_MAX_TIME="${MCTS_MAX_TIME:-0.04}"
MCTS_TIME_LEFT="${MCTS_TIME_LEFT:-5000}"
MCTS_MAX_MOVES="${MCTS_MAX_MOVES:-160}"
SELFPLAY_COMPRESS="${SELFPLAY_COMPRESS:-1}"
SELFPLAY_COMPRESSION_LEVEL="${SELFPLAY_COMPRESSION_LEVEL:-3}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model weights not found at $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -x "$STOCKFISH_BIN" ]]; then
  echo "Stockfish binary not found or not executable at $STOCKFISH_BIN" >&2
  exit 1
fi

mkdir -p "$SELFPLAY_ROOT"

while true; do
  stamp=$(date +%Y%m%d_%H%M%S)
  out_dir="$SELFPLAY_ROOT/$stamp"
  echo "[self-play] generating batch into $out_dir"
  args=(
    --model-path "$MODEL_PATH"
    --stockfish-path "$STOCKFISH_BIN"
    --stockfish-depth "$STOCKFISH_DEPTH"
    --stockfish-movetime "$STOCKFISH_MOVETIME"
    --output-dir "$out_dir"
    --games "$GAMES_PER_BATCH"
    --samples-per-shard "$SAMPLES_PER_SHARD"
    --policy-source "$POLICY_SOURCE"
    --value-source "$VALUE_SOURCE"
    --device cpu
    --workers "$WORKERS"
    --cpuct "$MCTS_CPUCT"
    --max-nodes "$MCTS_MAX_NODES"
    --max-time-seconds "$MCTS_MAX_TIME"
    --time-left-ms "$MCTS_TIME_LEFT"
    --max-moves "$MCTS_MAX_MOVES"
  )
  if [[ "$SELFPLAY_COMPRESS" == "1" ]]; then
    args+=(--compress --compression-level "$SELFPLAY_COMPRESSION_LEVEL")
  fi
  python3 "$BASE_DIR/training/self_play.py" "${args[@]}"

  echo "[self-play] uploading shards from $out_dir to Modal volume"
  modal volume put chess-data "$out_dir" "/selfplay/$stamp"
done
