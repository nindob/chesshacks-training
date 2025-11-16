from __future__ import annotations

import glob
import os
import io
from typing import Iterator, List

import torch
import zstandard as zstd

from training.data_pipeline import TrainingSample


def _serialize_sample(sample: TrainingSample, dtype: torch.dtype) -> dict:
    board_tensor = sample.board_tensor.to(dtype=dtype).cpu()
    move_tensors = {
        key: tensor.to(dtype=dtype if tensor.dtype.is_floating_point else tensor.dtype).cpu()
        for key, tensor in sample.move_tensors.items()
    }
    if isinstance(sample.value_target, torch.Tensor):
        value_target = sample.value_target.to(dtype=dtype).cpu()
    else:
        value_target = torch.tensor([float(sample.value_target)], dtype=dtype)
    policy_target = (
        sample.policy_target.to(dtype=dtype).cpu()
        if sample.policy_target is not None
        else None
    )
    return {
        "board_tensor": board_tensor,
        "move_tensors": move_tensors,
        "value_target": value_target,
        "policy_target": policy_target,
    }


def _deserialize_sample(entry: dict) -> TrainingSample:
    board_tensor = entry["board_tensor"].to(torch.float32)
    move_tensors = {
        key: tensor.to(torch.float32) if tensor.dtype.is_floating_point else tensor.long()
        for key, tensor in entry["move_tensors"].items()
    }
    value_target = entry["value_target"].to(torch.float32)
    policy_tensor = entry.get("policy_target")
    policy_target = policy_tensor.to(torch.float32) if policy_tensor is not None else None
    return TrainingSample(
        board_tensor=board_tensor,
        move_tensors=move_tensors,
        legal_moves=None,
        value_target=value_target,
        policy_target=policy_target,
    )


def save_shard(
    samples: List[TrainingSample],
    output_path: str,
    dtype: torch.dtype = torch.float16,
    compress: bool = False,
    compression_level: int = 3,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = [_serialize_sample(sample, dtype=dtype) for sample in samples]
    buffer = io.BytesIO()
    torch.save({"samples": payload}, buffer)
    data = buffer.getvalue()
    if compress:
        data = zstd.ZstdCompressor(level=compression_level).compress(data)
    with open(output_path, "wb") as handle:
        handle.write(data)


def _load_shard(path: str) -> List[TrainingSample]:
    with open(path, "rb") as handle:
        data = handle.read()
    if path.endswith(".zst"):
        data = zstd.ZstdDecompressor().decompress(data)
    payload = torch.load(io.BytesIO(data), map_location="cpu")
    entries = payload.get("samples") if isinstance(payload, dict) else payload
    return [_deserialize_sample(entry) for entry in entries]


def stream_tensor_shards(directory: str) -> Iterator[TrainingSample]:
    directory = os.path.abspath(directory)
    patterns = [os.path.join(directory, "*.pt"), os.path.join(directory, "*.pt.zst")]
    files = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not files:
        raise ValueError(f"No shard files found under {directory}")
    while True:
        for path in files:
            for sample in _load_shard(path):
                yield sample


__all__ = ["save_shard", "stream_tensor_shards"]
