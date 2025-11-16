from __future__ import annotations

import os

import modal

from training.config import DataPaths, TrainingConfig
from training.train import run_training

APP_NAME = "chesshacks-trainer"
VOLUME_NAME = "chess-data"
DATA_ROOT = "/data"
GPU_TYPE = os.environ.get("CHESS_MODAL_GPU", "T4")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("training/requirements.txt")
    .add_local_dir("training", "/root/training")
    .add_local_dir("src", "/root/src")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 12,
    volumes={DATA_ROOT: volume},
)
def train_on_modal(
    epochs: int = 2,
    steps_per_epoch: int = 4096,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    save_path: str = f"{DATA_ROOT}/weights/model.pt",
    init_weights: str | None = None,
    selfplay_dir: str | None = None,
    use_shards_only: bool = False,
):
    data_paths = DataPaths(
        chessdata_csv=os.path.join(DATA_ROOT, "chessData.csv"),
        random_evals_csv=os.path.join(DATA_ROOT, "random_evals.csv"),
        tactic_evals_csv=os.path.join(DATA_ROOT, "tactic_evals.csv"),
        lichess_elite_pgn=os.path.join(DATA_ROOT, "lichess_elite_2025-08.pgn"),
        lichess_rated_pgn=os.path.join(DATA_ROOT, "lichess_db_standard_rated_2025-09.pgn"),
        lichess_puzzle_csv=os.path.join(DATA_ROOT, "lichess_db_puzzle.csv"),
        selfplay_dir=selfplay_dir or os.path.join(DATA_ROOT, "selfplay"),
    )
    if use_shards_only:
        shards_path = data_paths.selfplay_dir
        data_paths = DataPaths(
            chessdata_csv=None,
            random_evals_csv=None,
            tactic_evals_csv=None,
            lichess_elite_pgn=None,
            lichess_rated_pgn=None,
            lichess_puzzle_csv=None,
            selfplay_dir=shards_path,
        )
    config = TrainingConfig(
        data_paths=data_paths,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        device="cuda",
        learning_rate=learning_rate,
        save_path=save_path,
        init_model_path=init_weights,
    )
    run_training(config)
