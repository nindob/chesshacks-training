from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import torch
import multiprocessing as mp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_PATH):
    path_str = str(path)
    if path_str not in os.sys.path:
        os.sys.path.append(path_str)

from src.engine.agent import AgentConfig, ChessAgent
from src.engine.encoding import BoardEncoder, MoveEncoder
from training.data_pipeline import TrainingSample
from training.tensor_shards import save_shard


def _policy_vector(legal_moves, target_move: Optional[chess.Move]) -> Optional[torch.Tensor]:
    if target_move is None:
        return None
    vec = torch.zeros((len(legal_moves),), dtype=torch.float32)
    for idx, move in enumerate(legal_moves):
        if move == target_move:
            vec[idx] = 1.0
            return vec
    return None


def _normalize_eval(cp: float) -> float:
    capped = max(min(cp, 2000), -2000)
    return math.tanh(capped / 600.0)


def build_sample(
    board: chess.Board,
    board_encoder: BoardEncoder,
    move_encoder: MoveEncoder,
    eval_value: float,
    policy_move: Optional[chess.Move],
) -> Optional[TrainingSample]:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    encoded = board_encoder.encode(board)
    move_tensors = move_encoder.encode(legal_moves, board)
    value = torch.tensor([_normalize_eval(eval_value)], dtype=torch.float32) if isinstance(eval_value, float) else eval_value
    policy_target = _policy_vector(legal_moves, policy_move)
    return TrainingSample(
        board_tensor=encoded.tensor,
        move_tensors=move_tensors,
        legal_moves=legal_moves,
        value_target=value,
        policy_target=policy_target,
    )


def play_self_play_games(args: argparse.Namespace, shard_prefix: str = ""):
    agent_device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    agent = ChessAgent(
        AgentConfig(
            model_path=args.model_path,
            device=agent_device,
            cpuct=args.cpuct,
            max_nodes=args.max_nodes,
            max_time_seconds=args.max_time_seconds,
        )
    )
    board_encoder = BoardEncoder(device=torch.device("cpu"))
    move_encoder = MoveEncoder(board_encoder, device=torch.device("cpu"))

    need_engine = (
        args.policy_source == "stockfish"
        or args.value_source == "stockfish"
    )
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path) if (need_engine and args.stockfish_path) else None
    if need_engine and engine is None:
        raise RuntimeError("Stockfish engine required but not available. Set --stockfish-path or change policy/value sources.")

    os.makedirs(args.output_dir, exist_ok=True)
    ext = ".pt.zst" if args.compress else ".pt"
    buffer: list[TrainingSample] = []
    pending_samples: list[tuple[TrainingSample, chess.Color]] = []
    shard_idx = 0
    progress = tqdm(range(args.games), desc="Self-play games")

    try:
        for _ in progress:
            board = chess.Board()
            halfmoves = 0
            pending_samples.clear()
            while not board.is_game_over() and halfmoves < args.max_moves:
                stockfish_cp = 0.0
                stockfish_move = None
                if engine:
                    if args.stockfish_movetime > 0:
                        limit = chess.engine.Limit(time=args.stockfish_movetime)
                    else:
                        limit = chess.engine.Limit(depth=args.stockfish_depth)
                    info = engine.analyse(board, limit)
                    score = info.get("score")
                    if score is not None:
                        stockfish_cp = score.pov(board.turn).score(mate_score=10000) or 0.0
                    pv = info.get("pv")
                    if pv:
                        stockfish_move = pv[0]

                move, _ = agent.select_move(board, time_left_ms=args.time_left_ms)
                policy_move = stockfish_move if args.policy_source == "stockfish" and stockfish_move else move
                eval_value = stockfish_cp if args.value_source == "stockfish" else 0.0
                current_turn = board.turn
                sample = build_sample(board, board_encoder, move_encoder, eval_value, policy_move)
                if sample:
                    if args.value_source == "stockfish":
                        buffer.append(sample)
                    else:
                        pending_samples.append((sample, current_turn))
                board.push(move)
                halfmoves += 1

                if len(buffer) >= args.samples_per_shard:
                    shard_path = os.path.join(args.output_dir, f"{shard_prefix}selfplay_{shard_idx:05d}{ext}")
                    save_shard(
                        buffer,
                        shard_path,
                        compress=args.compress,
                        compression_level=args.compression_level,
                    )
                    buffer.clear()
                    shard_idx += 1
        # finalize pending samples with game outcome
            if args.value_source != "stockfish" and pending_samples:
                outcome = board.outcome()
                for sample, sample_turn in pending_samples:
                    if outcome is None or outcome.winner is None:
                        value = 0.0
                    else:
                        value = 1.0 if outcome.winner == sample_turn else -1.0
                    sample.value_target[:] = value
                    buffer.append(sample)
                pending_samples.clear()
                if len(buffer) >= args.samples_per_shard:
                    shard_path = os.path.join(args.output_dir, f"{shard_prefix}selfplay_{shard_idx:05d}{ext}")
                    save_shard(
                        buffer,
                        shard_path,
                        compress=args.compress,
                        compression_level=args.compression_level,
                    )
                    buffer.clear()
                    shard_idx += 1
    finally:
        if engine:
            engine.quit()

    if buffer:
        shard_path = os.path.join(args.output_dir, f"{shard_prefix}selfplay_{shard_idx:05d}{ext}")
        save_shard(
            buffer,
            shard_path,
            compress=args.compress,
            compression_level=args.compression_level,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate self-play dataset shards.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the neural network weights.")
    parser.add_argument("--stockfish-path", type=str, default=None, help="Optional path to a Stockfish binary for supervision.")
    parser.add_argument("--stockfish-depth", type=int, default=10)
    parser.add_argument("--stockfish-movetime", type=float, default=0.0, help="Seconds per Stockfish analysis (overrides depth if >0).")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--samples-per-shard", type=int, default=2048)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--policy-source", choices=["stockfish", "agent"], default="stockfish")
    parser.add_argument("--value-source", choices=["stockfish", "outcome"], default="stockfish")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--cpuct", type=float, default=1.4)
    parser.add_argument("--max-nodes", type=int, default=96)
    parser.add_argument("--max-time-seconds", type=float, default=0.04)
    parser.add_argument("--time-left-ms", type=int, default=5000)
    parser.add_argument("--max-moves", type=int, default=160)
    parser.add_argument("--compress", action="store_true", help="Compress shard files with zstd.")
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel processes for self-play.")
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.policy_source == "stockfish" or args.value_source == "stockfish") and not args.stockfish_path:
        raise RuntimeError("Stockfish path required when using Stockfish for policy or value targets.")

    if args.workers <= 1:
        play_self_play_games(args)
        return

    processes = []
    games_per_worker = max(1, math.ceil(args.games / args.workers))
    for worker_id in range(args.workers):
        worker_args = argparse.Namespace(**vars(args))
        worker_args.games = games_per_worker
        p = mp.Process(target=play_self_play_games, args=(worker_args, f"w{worker_id:02d}_"))
        p.daemon = False
        p.start()
        processes.append(p)

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    main()
