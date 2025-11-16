from __future__ import annotations

import json
from typing import Dict, Optional
import traceback
from pathlib import Path

import chess
from huggingface_hub import hf_hub_download

from .engine.agent import ChessAgent, AgentConfig
from .utils import GameContext, chess_manager

# --------------------------------------------------------------------
# Hugging Face model configuration
#
# Make sure this matches what you created on the HF website:
#   - username:  rawrosauraus
#   - repo:      chesshacks-bot-v0       (or whatever you chose)
#   - filename:  model_v0.pt             (or weights/model_v0.pt)
# --------------------------------------------------------------------
HF_REPO_ID = "rawrosauraus/chesshacks-bot-v0"  # change if you used another name
HF_FILENAME = "model_v3.pt"                    # or "weights/model_v2.pt"
HF_CACHE_DIR = ".model_cache"                  # local cache for HF downloads
OPENING_BOOK_FILE = "data/opening_book.json"


def _init_agent() -> ChessAgent:
    """
    Initialize the ChessAgent with pretrained weights fetched from
    Hugging Face Hub.

    This is called once at import time so:
      - The model file is downloaded (or read from cache) once.
      - The same agent instance is reused for all moves.
    """
    try:
        ckpt_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            cache_dir=HF_CACHE_DIR,
        )

        config = AgentConfig(
            model_path=ckpt_path,
            # device=None lets ChessAgent auto-pick cuda/cpu.
            # You can force "cpu" here if you want:
            # device="cpu",
        )
        agent = ChessAgent(config=config)
        print(f"Initialized ChessAgent with HF weights at {ckpt_path}")
        return agent

    except Exception:
        # If something goes wrong (HF offline, bad repo id, etc.),
        # fall back to a baseline agent that uses default loading logic.
        traceback.print_exc()
        print("Falling back to ChessAgent() with default configuration.")
        return ChessAgent()


# Global agent instance reused for all moves
agent = _init_agent()
_opening_book: dict[str, str] = {}
_opening_book_path = Path(__file__).resolve().parents[1] / OPENING_BOOK_FILE
if _opening_book_path.exists():
    try:
        with _opening_book_path.open("r", encoding="utf-8") as f:
            _opening_book = json.load(f)
    except Exception:
        _opening_book = {}


def _safe_select_move(ctx: GameContext) -> chess.Move:
    """
    Calls agent.select_move and:
      - Logs the move probabilities,
      - Falls back to a simple uniform policy over legal moves if
        something crashes so games can still continue.
    """
    try:
        board = ctx.board
        book_move = _lookup_opening_move(board)
        if book_move:
            ctx.logProbabilities({book_move: 1.0})
            return book_move

        move, policy = agent.select_move(board, ctx.timeLeft)
        adjusted = _prefer_non_knight_move(board, policy)
        if adjusted is not None:
            move = adjusted
        ctx.logProbabilities(policy)
        return move
    except Exception:
        traceback.print_exc()
        legal_moves = list(ctx.board.legal_moves)
        if not legal_moves:
            # No legal moves -> let the framework handle game termination
            raise
        # Fallback: uniform distribution over legal moves
        fallback_policy = {move: 1 / len(legal_moves) for move in legal_moves}
        ctx.logProbabilities(fallback_policy)
        return legal_moves[0]


@chess_manager.entrypoint
def choose_move(ctx: GameContext):
    """
    Main entrypoint that ChessHacks devtools will call to get your bot's move.
    """
    return _safe_select_move(ctx)


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game starts; clear any last-move probabilities.
    """
    ctx.logProbabilities({})
def _lookup_opening_move(board: chess.Board) -> Optional[chess.Move]:
    if not _opening_book:
        return None
    fen = board.board_fen()
    if fen in _opening_book:
        try:
            return chess.Move.from_uci(_opening_book[fen])
        except ValueError:
            return None
    return None


def _prefer_non_knight_move(board: chess.Board, policy: Dict[chess.Move, float]) -> Optional[chess.Move]:
    if board.fullmove_number > 5 or not policy:
        return None
    ordered = sorted(policy.items(), key=lambda item: item[1], reverse=True)
    top_prob = ordered[0][1]
    threshold = top_prob * 0.75
    for candidate, prob in ordered[1:]:
        if prob < threshold:
            break
        piece = board.piece_at(candidate.from_square)
        if piece and piece.piece_type == chess.KNIGHT:
            continue
        return candidate
    return None
