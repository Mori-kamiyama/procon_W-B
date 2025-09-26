from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from data.greedy_dataset import enumerate_ops
from models.policy import ModelConfig, PolicyCNN, PolicyRetNet
from utils.io import load_problem, apply_ops
from utils.metrics import heuristic_value


@dataclass
class EvalResult:
    solved: bool
    steps: int
    ops: List[dict]
    final_h: int
    model_match: int


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta = payload["meta"]
    cfg = payload["config"]
    model_cfg = ModelConfig(
        size=meta["size"],
        vocab_size=meta["vocab_size"],
        num_ops=meta["num_ops"],
        embed_dim=cfg.get("embed_dim", 32),
    )
    arch = cfg.get("architecture", "cnn").lower()
    if arch == "cnn":
        model = PolicyCNN(
            model_cfg,
            hidden_channels=cfg.get("hidden_channels", 64),
            depth=cfg.get("depth", 3),
            dropout=cfg.get("dropout", 0.1),
        )
    elif arch == "retnet":
        model = PolicyRetNet(
            model_cfg,
            d_model=cfg.get("d_model", 128),
            num_layers=cfg.get("num_layers", 4),
            num_heads=cfg.get("num_heads", 4),
            dropout=cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown architecture {arch}")
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, meta


def teacher_pick(
    grid: List[List[int]], ops_catalog: List[dict], seen: set[tuple]
) -> tuple[int, List[List[int]]]:
    best_score = None
    best_idx = None
    best_grid = None
    for idx, op in enumerate(ops_catalog):
        trial = apply_ops(grid, [op])
        key = tuple(value for row in trial for value in row)
        if key in seen:
            continue
        score = heuristic_value(trial)
        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx
            best_grid = trial
    if best_idx is None:
        # As a fallback, allow revisiting even if it loops
        best_idx = 0
        best_grid = apply_ops(grid, [ops_catalog[best_idx]])
    return best_idx, best_grid


def run_policy(problem_path: Path, model: torch.nn.Module, device: torch.device, max_steps: int = 256) -> EvalResult:
    data = load_problem(problem_path)
    size = data["size"]
    grid = [row[:] for row in data["entities"]]
    ops_catalog = enumerate_ops(size)
    history: List[dict] = []
    match_count = 0
    seen = {tuple(value for row in grid for value in row)}

    for step in range(max_steps):
        h_val = heuristic_value(grid)
        if h_val == 0:
            return EvalResult(True, step, history, h_val, match_count)

        grid_tensor = torch.tensor(grid, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(grid_tensor)
            idx = int(torch.argmax(logits, dim=1).item())
        teacher_idx, teacher_grid = teacher_pick(grid, ops_catalog, seen)

        if idx == teacher_idx:
            chosen_idx = idx
            next_grid = teacher_grid
            match_count += 1
        else:
            chosen_idx = teacher_idx
            next_grid = teacher_grid

        history.append(dict(ops_catalog[chosen_idx]))
        grid = next_grid
        seen.add(tuple(value for row in grid for value in row))

    return EvalResult(False, max_steps, history, heuristic_value(grid), match_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy model on one puzzle")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--problem", type=Path, default=Path("problems/4x4/p000.json"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_steps", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, meta = build_model_from_checkpoint(args.checkpoint, device)
    res = run_policy(args.problem, model, device, max_steps=args.max_steps)

    print(f"problem: {args.problem}")
    print(f"solved: {res.solved}")
    print(f"steps: {res.steps}")
    print(f"final_h: {res.final_h}")
    if res.steps > 0:
        ratio = res.model_match / max(1, res.steps)
        print(f"model_match_steps: {res.model_match} ({ratio:.3f})")
    if res.ops:
        print("ops:")
        for i, op in enumerate(res.ops):
            print(f"  {i:03d}: {op}")


if __name__ == "__main__":
    main()
