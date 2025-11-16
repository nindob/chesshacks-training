ChessHacks — Engine Architecture & Development Notes

This document consolidates the requirements, constraints, and system design choices for our ChessHacks chess engine. It serves as persistent context for all code generation and technical decisions.

1. Competition Requirements
Core Rules

The engine must generate legal chess moves.

Move selection must fundamentally rely on a neural network.
If the neural network is removed and the bot still plays reasonably, it is disqualified.

No pre-trained chess models (LCZero nets, SF NNUE, AlphaZero nets, etc.).

No classical engines or heuristic evals in inference.

Training must be done by us, from scratch, using:

Lichess PGN data

Lichess puzzles

Any public FEN → Stockfish-eval dataset (allowed)

The Python entrypoint will be given a GameCtx:

board: python_chess.Board

timeleftWhite, timeleftBlack

Debug hooks (log, visualizeLogits)

We must return a legal python-chess Move.

Deployment Compute (Tournament)

CPU: 2 vCPUs

RAM: 32 GB

GPU: None OR Nvidia T4 (depending on match)

Time per move: ~150 ms budget

Entry must run under strict timeouts.

Training Compute (During Hackathon)

Modal GPUs:

T4, A10, A100, H100, B200 available

T4/A100 most realistic for fast turnaround

No restrictions on training size; only inference time & RAM matter.

2. Objectives
Primary Goal:

Produce a 2200–2600+ Elo engine under ChessHacks constraints.

Realistic plateau:

Baseline SL model: 1400–1700

Optimized CNN+MHSA+MoveHead: 1800–2200

SL + Self-play + fast MCTS: 2200–2500

Best-case execution: 2500+

Secondary / Optional Goal

Later, if time allows, produce a <10MB mini-model for the Knight’s Edge prize.

3. High-Level Architecture (Final)

We use a compact Residual CNN backbone, a single lightweight MHSA layer, and a variable-length MoveHead that scores only the legal moves. This avoids global 1858-move heads.

Input Encoding → 20 channels

12 piece planes (6 piece types × 2 colors)

Side-to-move

Castling rights (4 planes)

En passant file OR repetition/mobility channels (choose during implementation)

Always encode from the perspective of the player to move.

Backbone

8–12 residual blocks

Widths: 96–128

Early blocks: dilation = 1

Later blocks: dilation = 2

BatchNorm + ReLU

Entire backbone optimized for T4 inference.

Attention Layer

1 MHSA block over 8×8 grid tokens

Heads: 2–4

Purpose: long-range piece interactions

Keep this minimal to stay within time budget.

MoveHead (Key Component)

Instead of outputting a global 1858-vector, we:

Encode (from_square, to_square, promotion_piece) via embeddings

Pool global board context

Concatenate features

Score each legal move individually using an MLP

Scores are logits → choose best move or feed into MCTS.

Value Head

Small conv → flatten → MLP → scalar in [-1, 1]

4. Inference & Search (Tournament Logic)
Mandatory constraints:

One batched forward pass per move.
All legal successors → encode → stack → run model → pick best.

Search Algorithm

PUCT-style MCTS with shallow depth:

Depth ≈ 1.5–2 plies (cannot exceed 150 ms)

200–1000 nodes depending on clock

Must use neural network both for priors & leaf values

Without the network search becomes random → satisfies neural dependency rule

Time Budgeting

Use timeleftWhite/timeleftBlack for adaptive search width:

If low time: pure policy ranking (one forward pass)

If enough time: shallow MCTS loops until 150ms expired

Legal Safety

Always choose from board.legal_moves

No custom move generator required (python-chess handles it)

5. Training Pipeline
Phase 1 — Supervised Learning (Bootstrapping)

Data: Lichess month, elite month, puzzles, optionally FEN+SF-eval dump

Target:

Policy: actual move played (one-hot) OR soft targets from SF-eval dataset

Value: game outcome or engine eval

Loss:
L = CE(policy) + λ * MSE(value) + ε * entropy(policy)

Data augmentation: board flips, perspective switches

Phase 2 — Self-Play (Optional but strong)

Use our own MCTS engine to generate search-improved targets

MCTS visit-counts → policy targets

Final outcome → value target

1–2 iterations already give strong Elo gains

6. Expected Strength

Under 36 hours with T4/A100 training:

8–12 residual layers

1 MHSA

MoveHead

~2M–8M parameters

Shallow MCTS

Expected:

2000–2300 Elo baseline

2300–2500 with self-play

2600 possible but extremely tight