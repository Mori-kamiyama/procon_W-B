#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from utils.io import load_problem, apply_ops
from utils.metrics import heuristic_value

Grid = List[List[int]]
Op = Dict[str, int]


def enumerate_ops(size: int) -> List[Op]:
    ops: List[Op] = []
    for n in range(2, size + 1):
        for y in range(0, size - n + 1):
            for x in range(0, size - n + 1):
                ops.append({"x": x, "y": y, "n": n})
    return ops


def flatten(grid: Grid) -> Tuple[int, ...]:
    return tuple(value for row in grid for value in row)


def greedy_rollout(
    grid: Grid,
    ops: Sequence[Op],
    max_steps: int,
    sample_budget: int,
) -> Tuple[bool, List[Tuple[Grid, Op]]]:
    state: Grid = [row[:] for row in grid]
    visited = {flatten(state)}
    samples: List[Tuple[Grid, Op]] = []
    for _ in range(max_steps):
        if heuristic_value(state) == 0:
            return True, samples
        best_choice: Tuple[int, Op, Grid, Tuple[int, ...]] | None = None
        for op in ops:
            next_state = apply_ops(state, [op])
            key = flatten(next_state)
            if key in visited:
                continue
            h_val = heuristic_value(next_state)
            if best_choice is None or h_val < best_choice[0]:
                best_choice = (h_val, op, next_state, key)
        if best_choice is None:
            return False, samples
        _, op, next_state, key = best_choice
        samples.append(([row[:] for row in state], dict(op)))
        if sample_budget and len(samples) >= sample_budget:
            return False, samples
        state = next_state
        visited.add(key)
    return False, samples


def generate_dataset(
    problems_root: Path,
    size: int,
    max_samples: int,
    max_steps: int,
    output_path: Path,
) -> int:
    ops = enumerate_ops(size)
    collected = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out:
        for problem_path in sorted((problems_root / f"{size}x{size}").glob("*.json")):
            data = load_problem(problem_path)
            ok, samples = greedy_rollout(
                data["entities"],
                ops,
                max_steps=max_steps,
                sample_budget=max(0, max_samples - collected),
            )
            if not samples:
                continue
            for grid, op in samples:
                record = {
                    "problem": str(problem_path),
                    "grid": grid,
                    "move": op,
                }
                f_out.write(json.dumps(record, ensure_ascii=True) + "\n")
            collected += len(samples)
            if collected >= max_samples:
                break
            if not ok:
                # Stop early if this problem exhausted the step limit but we hit budget.
                continue
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Manhattan-greedy dataset")
    parser.add_argument("--problems_root", type=Path, default=Path("problems"))
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/datasets/manhattan_greedy.jsonl"))
    args = parser.parse_args()

    total = generate_dataset(
        problems_root=args.problems_root,
        size=args.size,
        max_samples=args.max_samples,
        max_steps=args.max_steps,
        output_path=args.output,
    )
    print(f"wrote {total} samples to {args.output}")


if __name__ == "__main__":
    main()
