from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils.io import apply_ops
from utils.metrics import heuristic_value

Op = Dict[str, int]
Grid = List[List[int]]


def enumerate_ops(size: int) -> List[Op]:
    ops: List[Op] = []
    for n in range(2, size + 1):
        for y in range(0, size - n + 1):
            for x in range(0, size - n + 1):
                ops.append({"x": x, "y": y, "n": n})
    return ops


def rotate_grid_ccw(grid: Tensor, k: int) -> Tensor:
    if k % 4 == 0:
        return grid
    return torch.rot90(grid, k=k % 4, dims=(0, 1))


def rotate_move_ccw(move: Op, size: int, k: int) -> Op:
    k = k % 4
    if k == 0:
        return dict(move)
    x, y, n = move["x"], move["y"], move["n"]
    if k == 1:
        return {"x": y, "y": size - x - n, "n": n}
    if k == 2:
        return {"x": size - x - n, "y": size - y - n, "n": n}
    if k == 3:
        return {"x": size - y - n, "y": x, "n": n}
    raise ValueError(f"Unexpected rotation k={k}")


def move_to_index(move: Op, ops: Sequence[Op]) -> int:
    target = (move["x"], move["y"], move["n"])
    for idx, op in enumerate(ops):
        if (op["x"], op["y"], op["n"]) == target:
            return idx
    raise ValueError(f"Move {move} not found in enumerated ops")


class GreedyTeacherDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        size: int,
        augment_rotations: bool = False,
        seed: int | None = None,
        indices: Sequence[int] | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.jsonl_path}")
        self.size = size
        self.ops = enumerate_ops(size)
        self.augment = augment_rotations
        self.rng = random.Random(seed)

        self.records: List[Tuple[Tensor, int]] = []
        self.max_value = 0
        tmp_records: List[Tuple[Tensor, int, Tensor]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                grid_list: Grid = data["grid"]
                move: Op = data["move"]
                grid_tensor = torch.tensor(grid_list, dtype=torch.long)
                self.max_value = max(self.max_value, int(grid_tensor.max().item()))
                label = move_to_index(move, self.ops)
                mask = self._compute_optimal_mask(grid_tensor)
                tmp_records.append((grid_tensor, label, mask))
        if indices is not None:
            self.records = [tmp_records[i] for i in indices]
        else:
            self.records = tmp_records
        if not self.records:
            raise ValueError(f"Dataset {self.jsonl_path} is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, Tensor]:
        grid, label, mask = self.records[idx]
        if not self.augment:
            return grid.clone(), label, mask.clone()
        k = self.rng.randrange(4)
        rotated_grid = rotate_grid_ccw(grid, k)
        move = self.ops[label]
        rotated_move = rotate_move_ccw(move, self.size, k)
        rotated_label = move_to_index(rotated_move, self.ops)
        rotated_mask = rotate_mask(mask, self.size, k, self.ops)
        return rotated_grid.clone(), rotated_label, rotated_mask

    @property
    def num_ops(self) -> int:
        return len(self.ops)

    @property
    def vocab_size(self) -> int:
        return self.max_value + 1

    def describe(self) -> Dict[str, int]:
        return {
            "size": self.size,
            "num_samples": len(self.records),
            "num_ops": self.num_ops,
            "vocab_size": self.vocab_size,
        }

    def _compute_optimal_mask(self, grid: Tensor) -> Tensor:
        best = None
        best_indices: List[int] = []
        grid_list = grid.tolist()
        for idx, op in enumerate(self.ops):
            new_grid = apply_ops(grid_list, [op])
            score = heuristic_value(new_grid)
            if best is None or score < best:
                best = score
                best_indices = [idx]
            elif score == best:
                best_indices.append(idx)
        mask = torch.zeros(len(self.ops), dtype=torch.float32)
        if best_indices:
            value = 1.0 / len(best_indices)
            mask[best_indices] = value
        else:
            mask += 1.0 / len(self.ops)
        return mask


def rotate_mask(mask: Tensor, size: int, k: int, ops: Sequence[Op]) -> Tensor:
    k = k % 4
    if k == 0:
        return mask.clone()
    rotated = torch.zeros_like(mask)
    for idx, op in enumerate(ops):
        rotated_op = rotate_move_ccw(op, size, k)
        ridx = move_to_index(rotated_op, ops)
        rotated[ridx] += mask[idx]
    return rotated
