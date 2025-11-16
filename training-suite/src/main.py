from __future__ import annotations

import traceback

import chess

from .engine import ChessAgent
from .utils import GameContext, chess_manager

agent = ChessAgent()


def _safe_select_move(ctx: GameContext) -> chess.Move:
    try:
        move, policy = agent.select_move(ctx.board, ctx.timeLeft)
        ctx.logProbabilities(policy)
        return move
    except Exception:
        traceback.print_exc()
        legal_moves = list(ctx.board.legal_moves)
        if not legal_moves:
            raise
        fallback_policy = {move: 1 / len(legal_moves) for move in legal_moves}
        ctx.logProbabilities(fallback_policy)
        return legal_moves[0]


@chess_manager.entrypoint
def choose_move(ctx: GameContext):
    return _safe_select_move(ctx)


@chess_manager.reset
def reset_func(ctx: GameContext):
    ctx.logProbabilities({})
