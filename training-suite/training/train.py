from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Dict, Optional, Tuple

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_PATH):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from src.engine.encoding import BoardEncoder, MoveEncoder
from src.engine.model import DualHeadChessModel, ModelConfig
from training.config import DataPaths, TrainingConfig, load_data_paths_from_env
from training.data_pipeline import (
    TrainingSample,
    create_data_sources,
    mixed_sample_stream,
)


def _compute_losses(
    model: DualHeadChessModel,
    sample: TrainingSample,
    device: torch.device,
    config: TrainingConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    board = sample.board_tensor.to(device)
    move_tensors = {k: v.to(device) for k, v in sample.move_tensors.items()}
    logits, value = model.evaluate_moves(board, move_tensors)
    stats: Dict[str, float] = {}

    target_value = sample.value_target.to(device)
    value_loss = F.mse_loss(value, target_value)
    loss = config.value_weight * value_loss
    stats["value_loss"] = float(value_loss.detach().cpu())

    if sample.policy_target is not None and sample.policy_target.shape[0] == logits.shape[0]:
        policy_target = sample.policy_target.to(device)
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -(policy_target * log_probs).sum()
        loss = loss + config.policy_weight * policy_loss
        stats["policy_loss"] = float(policy_loss.detach().cpu())

        entropy = -(torch.softmax(logits, dim=-1) * log_probs).sum()
        entropy_loss = -config.entropy_weight * entropy
        loss = loss + entropy_loss
        stats["entropy"] = float(entropy.detach().cpu())
    else:
        stats["policy_loss"] = 0.0
        stats["entropy"] = 0.0

    return loss, stats


def run_training(config: TrainingConfig):
    data_paths = config.data_paths
    existing_paths = data_paths.existing()
    if not existing_paths:
        raise RuntimeError("No dataset files found. Check the paths in training/config.py.")

    print("Using data sources:")
    for name, path in existing_paths.items():
        print(f" - {name}: {path}")

    device = torch.device(config.device)
    model = DualHeadChessModel(ModelConfig()).to(device)
    _maybe_load_initial_weights(model, config.init_model_path, device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    board_encoder = BoardEncoder(device=torch.device("cpu"))
    move_encoder = MoveEncoder(board_encoder, device=torch.device("cpu"))
    sources = create_data_sources(existing_paths, board_encoder, move_encoder)
    sample_stream = mixed_sample_stream(sources)

    global_step = 0
    for epoch in range(config.epochs):
        progress = tqdm(range(config.steps_per_epoch), desc=f"Epoch {epoch + 1}/{config.epochs}")
        epoch_policy = 0.0
        epoch_value = 0.0
        epoch_total = 0.0
        for _ in progress:
            optimizer.zero_grad()
            total_loss = 0.0
            metrics = {"value_loss": 0.0, "policy_loss": 0.0, "entropy": 0.0}
            for _ in range(config.batch_size):
                sample = next(sample_stream)
                loss, stats = _compute_losses(model, sample, device, config)
                loss = loss / config.batch_size
                loss.backward()
                total_loss += float(loss.detach().cpu())
                for key in metrics:
                    metrics[key] += stats.get(key, 0.0) / config.batch_size
            optimizer.step()
            global_step += 1
            progress.set_postfix(
                total=f"{total_loss:.3f}",
                value=f"{metrics['value_loss']:.3f}",
                policy=f"{metrics['policy_loss']:.3f}",
            )
            epoch_policy += metrics["policy_loss"]
            epoch_value += metrics["value_loss"]
            epoch_total += total_loss
        save_checkpoint(model, config.save_path)
        print(f"Saved checkpoint to {config.save_path}")
        record = {
            "epoch": epoch + 1,
            "avg_policy_loss": epoch_policy / config.steps_per_epoch,
            "avg_value_loss": epoch_value / config.steps_per_epoch,
            "avg_total_loss": epoch_total / config.steps_per_epoch,
        }
        _append_metrics(record, config.metrics_path)


def save_checkpoint(model: DualHeadChessModel, path: str):
    resolved = os.path.abspath(path)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    torch.save(model.state_dict(), resolved)


def _append_metrics(record: dict, path: str):
    resolved = os.path.abspath(path)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    if os.path.exists(resolved):
        with open(resolved, "r") as f:
            metrics = json.load(f)
    else:
        metrics = []
    metrics.append(record)
    with open(resolved, "w") as f:
        json.dump(metrics, f, indent=2)


def _maybe_load_initial_weights(model: DualHeadChessModel, init_path: Optional[str], device: torch.device):
    if not init_path:
        return
    resolved = os.path.abspath(init_path)
    if not os.path.exists(resolved):
        print(f"Init weights not found at {resolved}, training from scratch.")
        return
    try:
        state_dict = torch.load(resolved, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded initial weights from {resolved}")
        if missing:
            print(f" - missing keys: {missing}")
        if unexpected:
            print(f" - unexpected keys: {unexpected}")
    except Exception as exc:
        print(f"Failed to load initial weights from {resolved}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ChessHacks Dual-Head model.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Gradient steps per epoch.")
    parser.add_argument("--batch-size", type=int, default=None, help="Samples per optimization step.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Training device (cpu, cuda, cuda:0, ...).")
    parser.add_argument("--save-path", type=str, default=None, help="Where to store checkpoints.")
    parser.add_argument("--init-weights", type=str, default=None, help="Optional path to warm-start weights.")
    parser.add_argument("--chessdata", type=str, default=None, help="Path to chessData.csv")
    parser.add_argument("--random-evals", type=str, default=None, help="Path to random_evals.csv")
    parser.add_argument("--tactic-evals", type=str, default=None, help="Path to tactic_evals.csv")
    parser.add_argument("--lichess-elite-pgn", type=str, default=None, help="Path to lichess elite PGN")
    parser.add_argument("--lichess-rated-pgn", type=str, default=None, help="Path to lichess rated PGN")
    parser.add_argument("--lichess-puzzle", type=str, default=None, help="Path to lichess puzzle CSV")
    parser.add_argument("--selfplay-dir", type=str, default=None, help="Directory of tensor shard files generated via self-play.")
    return parser.parse_args()


def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    updates = {}
    for field in ["epochs", "steps_per_epoch", "batch_size", "learning_rate", "device", "save_path"]:
        arg_name = field if field != "learning_rate" else "lr"
        value = getattr(args, arg_name if arg_name != "lr" else "lr")
        if value is not None:
            key = field if field != "learning_rate" else "learning_rate"
            updates[key] = value
    if args.init_weights:
        updates["init_model_path"] = args.init_weights

    if updates:
        config = replace(config, **updates)

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
    config = replace(config, data_paths=data_paths)
    return config


def main():
    args = parse_args()
    config = TrainingConfig(data_paths=load_data_paths_from_env())
    config = apply_overrides(config, args)
    run_training(config)


if __name__ == "__main__":
    main()
