from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import chess
import torch


@dataclass(frozen=True)
class EncodedBoard:
    tensor: torch.Tensor
    turn: chess.Color


class BoardEncoder:
    """Encodes python-chess boards into 20-channel tensors."""

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.num_channels = 20

    def orient_square(self, square: chess.Square, turn: chess.Color) -> chess.Square:
        if turn == chess.WHITE:
            return square
        return chess.square_mirror(square)

    def encode(self, board: chess.Board) -> EncodedBoard:
        planes = torch.zeros(
            (self.num_channels, 8, 8),
            dtype=torch.float32,
            device=self.device,
        )

        turn = board.turn

        # Piece planes (12 total, friendly first)
        for square, piece in board.piece_map().items():
            oriented_square = self.orient_square(square, turn)
            rank = chess.square_rank(oriented_square)
            file = chess.square_file(oriented_square)
            row = 7 - rank
            col = file

            color_idx = 0 if piece.color == turn else 1
            plane_idx = color_idx * 6 + (piece.piece_type - 1)
            planes[plane_idx, row, col] = 1.0

        # Castling rights (4 planes: friendly/allied first)
        channel = 12
        friendly = turn
        enemy = not turn
        planes[channel] = (
            1.0 if board.has_kingside_castling_rights(friendly) else 0.0
        )
        planes[channel + 1] = (
            1.0 if board.has_queenside_castling_rights(friendly) else 0.0
        )
        planes[channel + 2] = (
            1.0 if board.has_kingside_castling_rights(enemy) else 0.0
        )
        planes[channel + 3] = (
            1.0 if board.has_queenside_castling_rights(enemy) else 0.0
        )

        # Side to move
        planes[16] = 1.0

        # En passant
        if board.ep_square is not None:
            oriented_ep = self.orient_square(board.ep_square, turn)
            rank = chess.square_rank(oriented_ep)
            file = chess.square_file(oriented_ep)
            row = 7 - rank
            col = file
            planes[17, row, col] = 1.0

        # Simple history features: halfmove clock + move count
        halfmove_value = min(board.halfmove_clock / 100.0, 1.0)
        move_count_value = min(board.fullmove_number / 100.0, 1.0)
        planes[18] = halfmove_value
        planes[19] = move_count_value

        return EncodedBoard(tensor=planes.unsqueeze(0), turn=turn)


class MoveEncoder:
    """Encodes move lists into tensors aligned with board orientation."""

    PROMOTION_IDS = {
        None: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }
    MATERIAL_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0,
    }
    MOVE_FEATURE_DIM = 6

    def __init__(self, board_encoder: BoardEncoder, device: torch.device | None = None):
        self.board_encoder = board_encoder
        self.device = device or torch.device("cpu")

    def _king_shield_score(self, board: chess.Board, color: chess.Color) -> float:
        king_sq = board.king(color)
        if king_sq is None:
            return 0.0
        coverage = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq] | chess.BB_SQUARES[king_sq])
        pawns = 0
        for sq in coverage:
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type == chess.PAWN:
                pawns += 1
        return pawns / 8.0

    def _king_center_score(self, board: chess.Board, color: chess.Color) -> float:
        king_sq = board.king(color)
        if king_sq is None:
            return 0.0
        rank = chess.square_rank(king_sq)
        file = chess.square_file(king_sq)
        distance = abs(rank - 3.5) + abs(file - 3.5)
        return 1.0 - (distance / 7.0)

    def _move_features(self, board: chess.Board, move: chess.Move) -> List[float]:
        moving_color = board.turn
        base_shield = self._king_shield_score(board, moving_color)
        base_center = self._king_center_score(board, moving_color)
        capture_piece = board.piece_at(move.to_square)
        capture_flag = 1.0 if capture_piece else 0.0
        capture_value = 0.0
        if capture_piece:
            capture_value = self.MATERIAL_VALUES.get(capture_piece.piece_type, 0.0) / 9.0
        gives_check = 1.0 if board.gives_check(move) else 0.0
        moving_piece = board.piece_at(move.from_square)
        king_move = 1.0 if moving_piece and moving_piece.piece_type == chess.KING else 0.0

        board.push(move)
        post_shield = self._king_shield_score(board, moving_color)
        post_center = self._king_center_score(board, moving_color)
        board.pop()

        shield_delta = post_shield - base_shield
        center_delta = post_center - base_center

        return [
            capture_flag,
            capture_value,
            gives_check,
            king_move,
            shield_delta,
            center_delta,
        ]

    def encode(self, moves: Iterable[chess.Move], board: chess.Board) -> dict[str, torch.Tensor]:
        move_list: List[chess.Move] = list(moves)
        if not move_list:
            return {
                "from": torch.zeros((0,), dtype=torch.long, device=self.device),
                "to": torch.zeros((0,), dtype=torch.long, device=self.device),
                "promotion": torch.zeros((0,), dtype=torch.long, device=self.device),
                "features": torch.zeros((0, self.MOVE_FEATURE_DIM), dtype=torch.float32, device=self.device),
            }

        turn = board.turn
        from_sq = []
        to_sq = []
        promotions = []
        feature_rows: List[List[float]] = []
        board_copy = board.copy(stack=False)

        for move in move_list:
            oriented_from = self.board_encoder.orient_square(move.from_square, turn)
            oriented_to = self.board_encoder.orient_square(move.to_square, turn)
            from_sq.append(oriented_from)
            to_sq.append(oriented_to)
            promotions.append(self.PROMOTION_IDS.get(move.promotion, 0))
            feature_rows.append(self._move_features(board_copy, move))

        return {
            "from": torch.tensor(from_sq, dtype=torch.long, device=self.device),
            "to": torch.tensor(to_sq, dtype=torch.long, device=self.device),
            "promotion": torch.tensor(promotions, dtype=torch.long, device=self.device),
            "features": torch.tensor(feature_rows, dtype=torch.float32, device=self.device),
        }
