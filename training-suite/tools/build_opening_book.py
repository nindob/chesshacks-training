from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict

import chess
import chess.pgn as chess_pgn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a simple opening book from PGN files.")
    parser.add_argument("--pgn-dir", required=True, help="Directory containing elite PGN files.")
    parser.add_argument("--output", required=True, help="Path to write the JSON opening book.")
    parser.add_argument("--depth", type=int, default=4, help="Number of plies to include.")
    parser.add_argument("--min-count", type=int, default=3, help="Minimum frequency for a move to be included.")
    return parser.parse_args()


def main():
    args = parse_args()
    entries: Dict[str, Counter] = defaultdict(Counter)

    processed_games = 0
    for fname in os.listdir(args.pgn_dir):
        if not fname.lower().endswith(".pgn"):
            continue
        path = os.path.join(args.pgn_dir, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            while True:
                game = chess_pgn.read_game(handle)
                if game is None:
                    break
                processed_games += 1
                if processed_games % 100 == 0:
                    print(f"Processed {processed_games} games...", flush=True)
                board = game.board()
                ply = 0
                for move in game.mainline_moves():
                    if ply >= args.depth:
                        break
                    fen = board.board_fen()
                    entries[fen][move.uci()] += 1
                    board.push(move)
                    ply += 1

    book = {}
    for fen, counter in entries.items():
        move, count = counter.most_common(1)[0]
        if count >= args.min_count:
            book[fen] = move

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(book, f, indent=2)


if __name__ == "__main__":
    main()
