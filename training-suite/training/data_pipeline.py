from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional

import chess
import chess.pgn as chess_pgn
import pandas as pd
import torch

from src.engine.encoding import BoardEncoder, MoveEncoder


@dataclass
class TrainingSample:
    board_tensor: torch.Tensor
    move_tensors: dict[str, torch.Tensor]
    legal_moves: Optional[List[chess.Move]]
    value_target: torch.Tensor
    policy_target: Optional[torch.Tensor]


def _normalize_value(eval_str: str, turn: chess.Color) -> float:
    text = str(eval_str).strip()
    if not text:
        return 0.0
    if text.startswith("#"):
        # Treat mate scores as ±1 depending on sign
        if "-" in text:
            return -1.0
        return 1.0
    try:
        centipawns = float(text)
    except ValueError:
        return 0.0
    signed_cp = centipawns if turn == chess.WHITE else -centipawns
    return math.tanh(signed_cp / 600.0)


def _policy_vector(legal_moves: List[chess.Move], target_move: Optional[chess.Move]) -> Optional[torch.Tensor]:
    if target_move is None:
        return None
    for idx, move in enumerate(legal_moves):
        if move == target_move:
            vec = torch.zeros((len(legal_moves),), dtype=torch.float32)
            vec[idx] = 1.0
            return vec
    return None


def _build_sample(
    board: chess.Board,
    board_encoder: BoardEncoder,
    move_encoder: MoveEncoder,
    value_str: str,
    policy_move: Optional[chess.Move] = None,
) -> Optional[TrainingSample]:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    encoded = board_encoder.encode(board)
    move_tensors = move_encoder.encode(legal_moves, board)
    value = torch.tensor([_normalize_value(value_str, board.turn)], dtype=torch.float32)
    policy = _policy_vector(legal_moves, policy_move)
    return TrainingSample(
        board_tensor=encoded.tensor,
        move_tensors=move_tensors,
        legal_moves=legal_moves,
        value_target=value,
        policy_target=policy,
    )


def _parse_move_str(move_str: str, board: chess.Board) -> Optional[chess.Move]:
    try:
        return board.parse_uci(move_str.strip())
    except ValueError:
        return None


def stream_value_csv(path: str, board_encoder: BoardEncoder, move_encoder: MoveEncoder, chunk_size: int = 4096) -> Iterator[TrainingSample]:
    columns = ["FEN", "Evaluation"]
    while True:
        for chunk in pd.read_csv(path, chunksize=chunk_size, usecols=columns):
            for _, row in chunk.iterrows():
                fen = row["FEN"]
                value = row["Evaluation"]
                try:
                    board = chess.Board(fen)
                except ValueError:
                    continue
                sample = _build_sample(board, board_encoder, move_encoder, str(value))
                if sample:
                    yield sample


def stream_tactics_csv(path: str, board_encoder: BoardEncoder, move_encoder: MoveEncoder, chunk_size: int = 4096) -> Iterator[TrainingSample]:
    columns = ["FEN", "Evaluation", "Move"]
    while True:
        for chunk in pd.read_csv(path, chunksize=chunk_size, usecols=columns):
            for _, row in chunk.iterrows():
                fen = row["FEN"]
                value = row["Evaluation"]
                move_str = row["Move"]
                if not isinstance(move_str, str):
                    continue
                try:
                    board = chess.Board(fen)
                except ValueError:
                    continue
                move = _parse_move_str(move_str, board)
                if move is None:
                    continue
                sample = _build_sample(board, board_encoder, move_encoder, str(value), move)
                if sample:
                    yield sample


def stream_puzzle_csv(path: str, board_encoder: BoardEncoder, move_encoder: MoveEncoder, chunk_size: int = 4096) -> Iterator[TrainingSample]:
    columns = ["FEN", "Moves"]
    while True:
        for chunk in pd.read_csv(path, chunksize=chunk_size, usecols=columns):
            for _, row in chunk.iterrows():
                fen = row["FEN"]
                moves_field = row["Moves"]
                if not isinstance(moves_field, str):
                    continue
                moves = moves_field.split()
                if not moves:
                    continue
                try:
                    board = chess.Board(fen)
                except ValueError:
                    continue
                move = _parse_move_str(moves[0], board)
                if move is None:
                    continue
                sample = _build_sample(board, board_encoder, move_encoder, "0", move)
                if sample:
                    yield sample


def _iter_pgn_files(path: str) -> List[str]:
    if os.path.isdir(path):
        return sorted(
            [
                os.path.join(path, fname)
                for fname in os.listdir(path)
                if fname.lower().endswith(".pgn")
            ]
        )
    return [path]


def stream_pgn_games(path: str, board_encoder: BoardEncoder, move_encoder: MoveEncoder) -> Iterator[TrainingSample]:
    def result_to_value(result: str, turn: chess.Color) -> float:
        if result == "1-0":
            return 1.0 if turn == chess.WHITE else -1.0
        if result == "0-1":
            return 1.0 if turn == chess.BLACK else -1.0
        return 0.0

    while True:
        file_list = _iter_pgn_files(path)
        if not file_list:
            raise ValueError(f"No PGN files found at {path}")
        for file_path in file_list:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                while True:
                    game = chess_pgn.read_game(handle)
                    if game is None:
                        break
                    result = game.headers.get("Result")
                    if result not in {"1-0", "0-1", "1/2-1/2"}:
                        continue
                    board = game.board()
                    for move in game.mainline_moves():
                        sample = _build_sample(
                            board,
                            board_encoder,
                            move_encoder,
                            str(result_to_value(result, board.turn) * 600),  # scale to cp range
                            move,
                        )
                        board.push(move)
                        if sample:
                            yield sample


@dataclass
class DataSource:
    name: str
    iterator_factory: Callable[[], Iterator[TrainingSample]]
    weight: float = 1.0


def create_data_sources(
    paths: Dict[str, str],
    board_encoder: BoardEncoder,
    move_encoder: MoveEncoder,
) -> List[DataSource]:
    sources: List[DataSource] = []
    if path := paths.get("chessdata_csv"):
        sources.append(
            DataSource(
                name="chessdata",
                iterator_factory=lambda p=path: stream_value_csv(p, board_encoder, move_encoder),
                weight=3.0,
            )
        )
    if path := paths.get("random_evals_csv"):
        sources.append(
            DataSource(
                name="random_evals",
                iterator_factory=lambda p=path: stream_value_csv(p, board_encoder, move_encoder),
                weight=2.0,
            )
        )
    if path := paths.get("tactic_evals_csv"):
        sources.append(
            DataSource(
                name="tactic_evals",
                iterator_factory=lambda p=path: stream_tactics_csv(p, board_encoder, move_encoder),
                weight=2.0,
            )
        )
    if path := paths.get("lichess_puzzle_csv"):
        sources.append(
            DataSource(
                name="lichess_puzzles",
                iterator_factory=lambda p=path: stream_puzzle_csv(p, board_encoder, move_encoder),
                weight=1.0,
            )
        )
    if path := paths.get("lichess_elite_pgn"):
        sources.append(
            DataSource(
                name="lichess_elite",
                iterator_factory=lambda p=path: stream_pgn_games(p, board_encoder, move_encoder),
                weight=2.5,
            )
        )
    if path := paths.get("lichess_rated_pgn"):
        sources.append(
            DataSource(
                name="lichess_rated",
                iterator_factory=lambda p=path: stream_pgn_games(p, board_encoder, move_encoder),
                weight=1.0,
            )
        )
    if path := paths.get("selfplay_dir"):
        from training.tensor_shards import stream_tensor_shards

        sources.append(
            DataSource(
                name="selfplay_shards",
                iterator_factory=lambda p=path: stream_tensor_shards(p),
                weight=2.0,
            )
        )
    return sources


def mixed_sample_stream(
    sources: List[DataSource],
    rng: Optional[random.Random] = None,
) -> Iterator[TrainingSample]:
    if not sources:
        raise ValueError("No data sources available. Check the dataset paths.")
    rng = rng or random.Random()
    iterators = [src.iterator_factory() for src in sources]
    weights = [src.weight for src in sources]

    while True:
        idx = rng.choices(range(len(sources)), weights=weights, k=1)[0]
        iterator = iterators[idx]
        try:
            yield next(iterator)
        except StopIteration:
            iterators[idx] = sources[idx].iterator_factory()


__all__ = [
    "TrainingSample",
    "create_data_sources",
    "mixed_sample_stream",
]
