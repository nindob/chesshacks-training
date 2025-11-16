from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch


def _expand(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(path))


DEFAULT_DOWNLOADS = os.path.expanduser("~/Downloads")


@dataclass
class DataPaths:
    chessdata_csv: Optional[str] = field(
        default_factory=lambda: os.path.join(DEFAULT_DOWNLOADS, "chessData.csv")
    )
    random_evals_csv: Optional[str] = field(
        default_factory=lambda: os.path.join(DEFAULT_DOWNLOADS, "random_evals.csv")
    )
    tactic_evals_csv: Optional[str] = field(
        default_factory=lambda: os.path.join(DEFAULT_DOWNLOADS, "tactic_evals.csv")
    )
    lichess_elite_pgn: Optional[str] = field(
        default_factory=lambda: os.path.join(DEFAULT_DOWNLOADS, "lichess_elite_2025-08.pgn")
    )
    lichess_rated_pgn: Optional[str] = field(
        default_factory=lambda: os.path.join(DEFAULT_DOWNLOADS, "lichess_db_standard_rated_2025-09.pgn")
    )
    lichess_puzzle_csv: Optional[str] = field(
        default_factory=lambda: os.path.join(DEFAULT_DOWNLOADS, "lichess_db_puzzle.csv")
    )
    selfplay_dir: Optional[str] = field(default=None)

    def resolved(self) -> "DataPaths":
        return DataPaths(
            chessdata_csv=_expand(self.chessdata_csv),
            random_evals_csv=_expand(self.random_evals_csv),
            tactic_evals_csv=_expand(self.tactic_evals_csv),
            lichess_elite_pgn=_expand(self.lichess_elite_pgn),
            lichess_rated_pgn=_expand(self.lichess_rated_pgn),
            lichess_puzzle_csv=_expand(self.lichess_puzzle_csv),
            selfplay_dir=_expand(self.selfplay_dir),
        )

    def as_dict(self) -> dict[str, Optional[str]]:
        return {
            "chessdata_csv": self.chessdata_csv,
            "random_evals_csv": self.random_evals_csv,
            "tactic_evals_csv": self.tactic_evals_csv,
            "lichess_elite_pgn": self.lichess_elite_pgn,
            "lichess_rated_pgn": self.lichess_rated_pgn,
            "lichess_puzzle_csv": self.lichess_puzzle_csv,
            "selfplay_dir": self.selfplay_dir,
        }

    def existing(self) -> dict[str, str]:
        resolved = self.resolved()
        return {
            name: path
            for name, path in resolved.as_dict().items()
            if path and os.path.exists(path)
        }


def load_data_paths_from_env() -> DataPaths:
    return DataPaths(
        chessdata_csv=_expand(
            os.environ.get("CHESSDATA_CSV_PATH", DataPaths().chessdata_csv)
        ),
        random_evals_csv=_expand(
            os.environ.get("RANDOM_EVALS_CSV_PATH", DataPaths().random_evals_csv)
        ),
        tactic_evals_csv=_expand(
            os.environ.get("TACTIC_EVALS_CSV_PATH", DataPaths().tactic_evals_csv)
        ),
        lichess_elite_pgn=_expand(
            os.environ.get("LICHESS_ELITE_PGN_PATH", DataPaths().lichess_elite_pgn)
        ),
        lichess_rated_pgn=_expand(
            os.environ.get("LICHESS_RATED_PGN_PATH", DataPaths().lichess_rated_pgn)
        ),
        lichess_puzzle_csv=_expand(
            os.environ.get("LICHESS_PUZZLE_CSV_PATH", DataPaths().lichess_puzzle_csv)
        ),
        selfplay_dir=_expand(
            os.environ.get("SELFPLAY_SHARDS_DIR")
        ),
    )


@dataclass
class TrainingConfig:
    data_paths: DataPaths = field(default_factory=load_data_paths_from_env)
    epochs: int = 2
    steps_per_epoch: int = 4096
    batch_size: int = 64
    value_weight: float = 1.0
    policy_weight: float = 1.0
    entropy_weight: float = 1e-4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = field(
        default_factory=lambda: os.path.join("src", "weights", "model.pt")
    )
    init_model_path: Optional[str] = None
    metrics_path: str = field(
        default_factory=lambda: os.path.join("artifacts", "training_metrics.json")
    )


__all__ = [
    "DataPaths",
    "TrainingConfig",
    "load_data_paths_from_env",
]
