from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_PATH):
    path_str = str(path)
    if path_str not in os.sys.path:
        os.sys.path.append(path_str)

from src.engine.encoding import BoardEncoder, MoveEncoder
from training.config import TrainingConfig, load_data_paths_from_env
from training.data_pipeline import create_data_sources, mixed_sample_stream
from training.tensor_shards import save_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute tensor shards for faster training.")
    parser.add_argument("--output-dir", required=True, help="Directory to store shard files.")
    parser.add_argument("--samples-per-shard", type=int, default=4096)
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Precision used when saving floating tensors.",
    )
    parser.add_argument("--chessdata", type=str, default=None, help="Override chessData.csv path.")
    parser.add_argument("--random-evals", type=str, default=None, help="Override random_evals.csv path.")
    parser.add_argument("--tactic-evals", type=str, default=None, help="Override tactic_evals.csv path.")
    parser.add_argument("--lichess-elite-pgn", type=str, default=None, help="Override lichess elite PGN path.")
    parser.add_argument("--lichess-rated-pgn", type=str, default=None, help="Override lichess rated PGN path.")
    parser.add_argument("--lichess-puzzle", type=str, default=None, help="Override lichess puzzle CSV path.")
    parser.add_argument("--selfplay-dir", type=str, default=None, help="Directory containing self-play shards.")
    parser.add_argument("--compress", action="store_true", help="Compress shard files with zstd.")
    parser.add_argument("--compression-level", type=int, default=3, help="zstd compression level.")
    return parser.parse_args()


def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    data_paths = config.data_paths
    overrides = {
        "chessdata_csv": args.chessdata,
        "random_evals_csv": args.random_evals,
        "tactic_evals_csv": args.tactic_evals,
        "lichess_elite_pgn": args.lichess_elite_pgn,
        "lichess_rated_pgn": args.lichess_rated_pgn,
        "lichess_puzzle_csv": args.lichess_puzzle,
        "selfplay_dir": args.selfplay_dir,
    }
    for attr, value in overrides.items():
        if value:
            setattr(data_paths, attr, value)
    return replace(config, data_paths=data_paths)


def main():
    args = parse_args()
    config = TrainingConfig(data_paths=load_data_paths_from_env())
    config = apply_overrides(config, args)

    existing_paths = config.data_paths.existing()
    if not existing_paths:
        raise RuntimeError("No datasets available. Provide at least one path.")

    board_encoder = BoardEncoder(device=torch.device("cpu"))
    move_encoder = MoveEncoder(board_encoder, device=torch.device("cpu"))
    sources = create_data_sources(existing_paths, board_encoder, move_encoder)
    sampler = mixed_sample_stream(sources)

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    ext = ".pt.zst" if args.compress else ".pt"
    total_samples = args.samples_per_shard * args.num_shards
    progress = tqdm(range(total_samples), desc="Generating shards")
    buffer = []
    shard_idx = 0

    for _ in progress:
        buffer.append(next(sampler))
        if len(buffer) >= args.samples_per_shard:
            shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}{ext}")
            save_shard(
                buffer,
                shard_path,
                dtype=dtype,
                compress=args.compress,
                compression_level=args.compression_level,
            )
            buffer.clear()
            shard_idx += 1
    if buffer:
        shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}{ext}")
        save_shard(
            buffer,
            shard_path,
            dtype=dtype,
            compress=args.compress,
            compression_level=args.compression_level,
        )


if __name__ == "__main__":
    main()
