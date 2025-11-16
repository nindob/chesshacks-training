from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return F.relu(out, inplace=True)


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 20
    channels: int = 128
    residual_blocks: int = 8
    attn_heads: int = 4
    move_embedding_dim: int = 64
    move_head_hidden: int = 192
    value_hidden: int = 128
    move_feature_dim: int = 6
    heuristic_scale: float = 0.08


class MoveHead(nn.Module):
    def __init__(self, channels: int, config: ModelConfig):
        super().__init__()
        embed_dim = config.move_embedding_dim
        self.from_embed = nn.Embedding(64, embed_dim)
        self.to_embed = nn.Embedding(64, embed_dim)
        self.promo_embed = nn.Embedding(5, embed_dim)
        total_dim = channels * 3 + embed_dim * 3 + config.move_feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, config.move_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.move_head_hidden, 1),
        )
        self.heuristic_scale = config.heuristic_scale

    def forward(
        self,
        tokens: torch.Tensor,
        global_context: torch.Tensor,
        move_tensors: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch, _, channels = tokens.shape
        if batch != 1:
            raise ValueError("MoveHead currently expects batch size of 1")

        move_count = move_tensors["from"].shape[0]
        if move_count == 0:
            return torch.empty((0,), device=tokens.device)

        token_bank = tokens[0]
        from_features = token_bank[move_tensors["from"]]
        to_features = token_bank[move_tensors["to"]]
        global_features = global_context[0].expand(move_count, -1)

        move_features = torch.cat(
            [
                global_features,
                from_features,
                to_features,
                self.from_embed(move_tensors["from"]),
                self.to_embed(move_tensors["to"]),
                self.promo_embed(move_tensors["promotion"]),
                move_tensors["features"],
            ],
            dim=-1,
        )

        logits = self.mlp(move_features).squeeze(-1)
        if self.heuristic_scale > 0:
            logits = logits + self.heuristic_scale * self._heuristic_scores(move_tensors["features"])
        return logits

    def _heuristic_scores(self, features: torch.Tensor) -> torch.Tensor:
        capture_flag = features[:, 0]
        capture_value = features[:, 1]
        gives_check = features[:, 2]
        king_move = features[:, 3]
        shield_delta = features[:, 4]
        center_delta = features[:, 5]

        positive = (
            0.05 * capture_flag
            + 0.08 * capture_value
            + 0.04 * gives_check
            + 0.05 * torch.relu(shield_delta)
            + 0.04 * torch.relu(center_delta)
        )
        negative = (
            0.12 * king_move
            + 0.08 * torch.relu(-shield_delta)
            + 0.05 * torch.relu(-center_delta)
        )
        return positive - negative


class DualHeadChessModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(config.input_channels, config.channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.channels),
            nn.ReLU(inplace=True),
        )

        blocks = []
        for block_idx in range(config.residual_blocks):
            dilation = 2 if block_idx >= config.residual_blocks // 2 else 1
            blocks.append(ResidualBlock(config.channels, dilation=dilation))
        self.residual_stack = nn.Sequential(*blocks)

        self.tokens_norm = nn.LayerNorm(config.channels)
        self.positional = nn.Parameter(torch.zeros(64, config.channels))
        self.attention = nn.MultiheadAttention(
            config.channels, config.attn_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(config.channels)

        self.value_head = nn.Sequential(
            nn.Linear(config.channels, config.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )

        self.move_head = MoveHead(config.channels, config)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.positional, std=0.02)

    def _extract_tokens(self, board_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(board_tensor)
        features = self.residual_stack(features)
        tokens = features.flatten(2).transpose(1, 2)  # (B, 64, C)
        tokens = self.tokens_norm(tokens)
        tokens = tokens + self.positional.unsqueeze(0)
        attn_out, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + attn_out
        tokens = self.attn_norm(tokens)
        global_context = tokens.mean(dim=1, keepdim=True)
        return tokens, global_context

    def evaluate_moves(
        self,
        board_tensor: torch.Tensor,
        move_tensors: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, global_context = self._extract_tokens(board_tensor)
        move_logits = self.move_head(tokens, global_context, move_tensors)
        value = self.value_head(global_context.squeeze(1))
        return move_logits, value.squeeze(-1)
