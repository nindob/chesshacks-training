from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import chess
import torch
import urllib.request
import urllib.parse
import pathlib

from .encoding import BoardEncoder, MoveEncoder
from .model import DualHeadChessModel, ModelConfig
from .mcts import MCTSConfig, NeuralMCTS


@dataclass
class AgentConfig:
    # If provided, model_path is used directly. Otherwise we fall back to:
    #   - CHESS_MODEL_URL (download to cache), or
    #   - CHESS_MODEL_PATH, or
    #   - src/../weights/model.pt
    model_path: Optional[str] = None
    # "cuda", "cpu", or None to auto-select.
    device: Optional[str] = None
    cpuct: float = 1.4
    max_nodes: int = 512
    max_time_seconds: float = 0.12


class ChessAgent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

        requested_device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(requested_device)

        self.board_encoder = BoardEncoder(self.device)
        self.move_encoder = MoveEncoder(self.board_encoder, self.device)
        self.model = DualHeadChessModel(ModelConfig()).to(self.device)
        self.model.eval()

        self._load_weights(self.config.model_path)

        mcts_config = MCTSConfig(
            cpuct=self.config.cpuct,
            max_nodes=self.config.max_nodes,
            max_time_seconds=self.config.max_time_seconds,
        )
        self.mcts = NeuralMCTS(self._evaluate_board, config=mcts_config)

    def _load_weights(self, model_path: Optional[str]):
        """
        Load model weights from one of:
          1. CHESS_MODEL_URL (downloaded to CHESS_MODEL_CACHE / .model_cache), or
          2. model_path passed via AgentConfig, or
          3. CHESS_MODEL_PATH env var, or
          4. local src/../weights/model.pt

        If none exists, the agent runs with randomly initialized weights.
        """
        url = os.environ.get("CHESS_MODEL_URL")
        cache_dir = os.environ.get("CHESS_MODEL_CACHE", "./.model_cache")

        if url:
            path = self._download_weights(url, cache_dir)
        else:
            path = (
                model_path
                or os.environ.get("CHESS_MODEL_PATH")
                or os.path.join(
                    os.path.dirname(__file__), "..", "weights", "model.pt"
                )
            )
            path = os.path.abspath(path)

        if not os.path.exists(path):
            print(f"No weights found at {path}, using randomly initialized model.")
            return

        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model weights from {path}")
        except Exception as exc:
            print(f"Failed to load weights at {path}: {exc}")

    def _download_weights(self, url: str, cache_dir: str) -> str:
        """
        Download weights from a URL (e.g., direct file link) into a cache directory.
        If the file already exists in the cache, reuse it.
        """
        cache_path = pathlib.Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        filename = pathlib.Path(urllib.parse.urlparse(url).path).name or "model.pt"
        target = cache_path / filename

        if target.exists():
            return str(target)

        tmp_path = target.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.replace(target)
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to download weights from {url}: {exc}")

        return str(target)

    def select_move(
        self, board: chess.Board, time_left_ms: Optional[int]
    ) -> Tuple[chess.Move, Dict[chess.Move, float]]:
        time_left = time_left_ms if time_left_ms is not None else 1000
        move, policy = self.mcts.search(board, time_left)
        return move, policy

    def _evaluate_board(
        self, board: chess.Board
    ) -> Tuple[Dict[chess.Move, float], float]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, 0.0

        encoded = self.board_encoder.encode(board)
        move_tensors = self.move_encoder.encode(legal_moves, board)
        if move_tensors["from"].shape[0] == 0:
            return {}, 0.0

        with torch.no_grad():
            logits, value = self.model.evaluate_moves(encoded.tensor, move_tensors)

        priors_tensor = torch.softmax(logits, dim=-1)
        priors = {
            move: float(prob)
            for move, prob in zip(legal_moves, priors_tensor.cpu().tolist())
        }
        return priors, float(value.item())
