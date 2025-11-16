from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Tuple

import chess

PolicyValueFn = Callable[[chess.Board], Tuple[Dict[chess.Move, float], float]]


@dataclass
class MCTSConfig:
    cpuct: float = 1.4
    max_nodes: int = 512
    max_time_seconds: float = 0.15
    min_simulations: int = 64
    fast_policy_only_time_ms: int = 75
    temperature: float = 1e-3
    high_time_threshold_ms: int = 40000
    high_time_max_nodes: int = 512
    high_time_max_seconds: float = 0.15
    medium_time_threshold_ms: int = 15000
    medium_time_max_nodes: int = 256
    medium_time_max_seconds: float = 0.09
    blunder_threshold: float = -0.8


class TreeNode:
    def __init__(
        self,
        board: chess.Board,
        prior: float = 1.0,
        parent: Optional["TreeNode"] = None,
        move: Optional[chess.Move] = None,
    ):
        self.board = board
        self.prior = prior
        self.parent = parent
        self.move = move
        self.children: Dict[chess.Move, TreeNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return self.is_expanded


class NeuralMCTS:
    def __init__(self, evaluator: PolicyValueFn, config: Optional[MCTSConfig] = None):
        self.evaluator = evaluator
        self.config = config or MCTSConfig()

    def search(self, board: chess.Board, time_left_ms: int) -> Tuple[chess.Move, Dict[chess.Move, float]]:
        max_nodes = self.config.max_nodes
        max_time = self.config.max_time_seconds
        if time_left_ms >= self.config.high_time_threshold_ms:
            max_nodes = self.config.high_time_max_nodes
            max_time = self.config.high_time_max_seconds
        elif time_left_ms >= self.config.medium_time_threshold_ms:
            max_nodes = self.config.medium_time_max_nodes
            max_time = self.config.medium_time_max_seconds
        elif time_left_ms <= self.config.fast_policy_only_time_ms:
            max_nodes = 0
            max_time = 0.0

        root = TreeNode(board.copy(stack=False))
        priors, value = self.evaluator(root.board)
        root.is_expanded = True
        if not priors:
            return self._fallback_move(root.board), {}

        total_prior = sum(priors.values())
        if total_prior <= 0:
            priors = {move: 1.0 for move in priors}
            total_prior = len(priors)
        for move, prior in priors.items():
            next_board = root.board.copy(stack=False)
            next_board.push(move)
            root.children[move] = TreeNode(
                board=next_board,
                prior=prior / total_prior,
                parent=root,
                move=move,
            )

        root.value_sum = value
        root.visit_count = 1

        if max_nodes == 0:
            best = self._select_best_move(root)
            best = self._blunder_filter(root, best)
            return best, self._policy_from_visits(root)

        start_time = time.time()
        simulations = 0
        while simulations < max_nodes:
            if (time.time() - start_time) > max_time:
                break
            node = root
            path = [node]

            # Selection
            while node.expanded() and node.children:
                node = self._select_child(node)
                path.append(node)

            leaf_value = self._expand(node)
            self._backpropagate(path, leaf_value)
            simulations += 1

            if simulations < self.config.min_simulations:
                continue

        best_move = self._select_best_move(root)
        best_move = self._blunder_filter(root, best_move)
        return best_move, self._policy_from_visits(root)

    def _blunder_filter(self, root: TreeNode, move: chess.Move) -> chess.Move:
        if move is None or self.config.blunder_threshold is None:
            return move
        board = root.board.copy(stack=False)
        board.push(move)
        priors, value = self.evaluator(board)
        if value < self.config.blunder_threshold and root.children:
            alt = max(
                (child for child in root.children.values() if child.move != move),
                key=lambda child: child.prior,
                default=None,
            )
            if alt:
                return alt.move
        return move

    def _select_child(self, node: TreeNode) -> TreeNode:
        assert node.children, "Cannot select a child from a leaf node"
        total_visits = sum(child.visit_count for child in node.children.values()) + 1
        best_score = -math.inf
        best_child = None
        for move, child in node.children.items():
            exploration = (
                self.config.cpuct
                * child.prior
                * math.sqrt(node.visit_count + 1)
                / (1 + child.visit_count)
            )
            score = child.q_value + exploration
            if score > best_score:
                best_score = score
                best_child = child

        return best_child  # type: ignore[return-value]

    def _expand(self, node: TreeNode) -> float:
        if node.board.is_game_over():
            outcome = node.board.outcome()
            if outcome is None or outcome.winner is None:
                return 0.0
            return 1.0 if outcome.winner == node.board.turn else -1.0

        priors, value = self.evaluator(node.board)
        if not priors:
            node.is_expanded = True
            return value

        total_prior = sum(priors.values())
        if total_prior <= 0:
            total_prior = 1.0
        for move, prior in priors.items():
            next_board = node.board.copy(stack=False)
            next_board.push(move)
            node.children.setdefault(
                move,
                TreeNode(
                    board=next_board,
                    prior=prior / total_prior,
                    parent=node,
                    move=move,
                ),
            )

        node.is_expanded = True
        return value

    def _backpropagate(self, path: Iterable[TreeNode], leaf_value: float):
        value = leaf_value
        for node in reversed(list(path)):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _select_best_move(self, root: TreeNode) -> chess.Move:
        if not root.children:
            return self._fallback_move(root.board)

        visits = [child.visit_count for child in root.children.values()]
        if sum(visits) == 0:
            return max(root.children.values(), key=lambda child: child.prior).move  # type: ignore[arg-type]

        return max(root.children.values(), key=lambda child: child.visit_count).move  # type: ignore[arg-type]

    def _policy_from_visits(self, root: TreeNode) -> Dict[chess.Move, float]:
        if not root.children:
            return {}
        visits = {child.move: child.visit_count for child in root.children.values()}
        total = sum(visits.values())
        if total <= 0:
            prior_sum = sum(child.prior for child in root.children.values())
            if prior_sum <= 0:
                return {child.move: 1 / len(root.children) for child in root.children.values()}
            return {child.move: child.prior / prior_sum for child in root.children.values()}
        return {move: count / total for move, count in visits.items()}

    def _fallback_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        return legal_moves[0]
