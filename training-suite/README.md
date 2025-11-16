# Training & Data Pipeline

This directory keeps all scripts that touch the big offline datasets (Kaggle Stockfish evals, random positions, tactical lines, and Lichess archives) plus the Modal entrypoint we use for GPU training. Nothing here is imported by the tournament bot.

## Data locations

All raw files currently live under `~/Downloads` on Ani’s machine:

| Dataset | Default path | Notes |
| --- | --- | --- |
| Kaggle Stockfish evals | `~/Downloads/chessData.csv` | FEN + SF11 @ depth 22 |
| Kaggle random evals | `~/Downloads/random_evals.csv` | FEN + SF12 NNUE-off |
| Kaggle tactic evals | `~/Downloads/tactic_evals.csv` | FEN + eval + best move |
| Lichess elite games | `~/Downloads/lichess_elite_2025-08.pgn` | unzip from `.zip` |
| Lichess rated games | `~/Downloads/lichess_db_standard_rated_2025-09.pgn` | decompress `.zst` |
| Lichess puzzles | `~/Downloads/lichess_db_puzzle.csv` | decompress `.zst` |

Update the environment variables in `training/config.py` if you relocate any of the files. Every script calls `expanduser` so `~` works out of the box.

### Decompress the `.zst` archives

macOS doesn’t register `.zst` files, so run our helper instead of double‑clicking:

```bash
python training/tools/decompress_zst.py \
  --input ~/Downloads/lichess_db_standard_rated_2025-09.pgn.zst \
  --output ~/Downloads/lichess_db_standard_rated_2025-09.pgn
```

Repeat for the puzzle CSV. The script streams the file so it handles the 80–90 M game dumps fine.

## Local preprocessing & training

Install the training dependencies (kept separate from the tournament requirements):

```bash
python -m venv .train-venv
source .train-venv/bin/activate
pip install -r training/requirements.txt
```

Kick off a quick sanity training loop on CPU/GPU:

```bash
python -m training.train \
  --save-path ./artifacts/model.pt \
  --steps-per-epoch 2000 \
  --epochs 1
```

Flags override the defaults in `TrainingConfig`. The script automatically samples from every dataset whose path exists. Value‑only sources (e.g., `chessData.csv`) train the value head; sources with best moves (tactics, PGNs, puzzles) also supervise the policy head.

## Modal GPU workflow

We run the heavy training on Modal’s cloud GPUs.

1. Upload datasets to a Modal Volume once (do this locally):
   ```bash
   modal volume create chess-data
   modal volume put chess-data ~/Downloads/chessData.csv
   modal volume put chess-data ~/Downloads/random_evals.csv
   # ...repeat for every file you want accessible on Modal
   ```
2. Update `DATA_ROOT` in `training/modal_entry.py` if you mounted the volume elsewhere.
3. Launch a job:
   ```bash
   modal run training.modal_entry::train_on_modal --epochs 4 --save-path /data/weights/latest.pt
   ```

The stub builds an image from `training/requirements.txt`, mounts the `chess-data` volume at `/data`, and calls the same `training.train` module. Checkpoints are written inside the mounted volume so you can `modal volume get` them afterwards.

## File overview

| File | Purpose |
| --- | --- |
| `config.py` | Dataclasses + helpers that centralize dataset paths and hyper‑parameters |
| `data_pipeline.py` | Streaming iterators for CSV/Puzzle/PGN data and normalized targets |
| `train.py` | PyTorch training loop that supervises both heads and saves checkpoints |
| `modal_entry.py` | Modal stub + CLI for launching GPU training jobs |
| `tools/decompress_zst.py` | CLI helper to unpack `.zst` archives without Finder |

Keep all heavy experiments here so the production bot stays lean. Once you’re happy with a checkpoint, either upload it to remote storage (Hugging Face Hub, S3, Modal volume, etc.) or copy a <100 MB file into the bot repo’s `src/weights/model.pt`. The deployed bot always reads from `CHESS_MODEL_PATH` (remote) or the local weights directory, so no training assets need to ship with the competition code.
