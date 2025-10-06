#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.run_search import ensure_cli_solver, run_one_problem
from src.utils.heuristic_config import reload_heuristic_config
from src.utils.io import list_problem_files


@dataclass
class TrialParams:
    weight_euclidean: float
    weight_chebyshev: float
    offset_euclidean: float
    offset_chebyshev: float
    epsilon: float


@dataclass
class TrialResult:
    index: int
    params: TrialParams
    solved: int
    total: int
    avg_h: float
    avg_time: float
    metrics: List[Dict[str, Any]]

    @property
    def solved_rate(self) -> float:
        return 0.0 if self.total == 0 else self.solved / float(self.total)


def _parse_range(values: Sequence[float], name: str) -> Tuple[float, float]:
    if len(values) == 1:
        values = (values[0], values[0])
    if len(values) != 2:
        raise ValueError(f"Expected two floats for {name}, got {values}")
    low, high = float(values[0]), float(values[1])
    if low > high:
        low, high = high, low
    return low, high


def sample_params(rng: random.Random, args: argparse.Namespace) -> TrialParams:
    w_e = rng.uniform(*args.weight_euclid_range)
    w_c = rng.uniform(*args.weight_cheby_range)
    if args.normalize_weights:
        total = w_e + w_c
        if total > 0:
            w_e /= total
            w_c /= total
    offset_e = rng.uniform(*args.offset_euclid_range)
    offset_c = rng.uniform(*args.offset_cheby_range)
    return TrialParams(
        weight_euclidean=w_e,
        weight_chebyshev=w_c,
        offset_euclidean=offset_e,
        offset_chebyshev=offset_c,
        epsilon=args.epsilon,
    )


def write_config(path: Path, params: TrialParams) -> None:
    lines = [
        "weights:",
        f"  euclidean: {params.weight_euclidean}",
        f"  chebyshev: {params.weight_chebyshev}",
        "offsets:",
        f"  euclidean: {params.offset_euclidean}",
        f"  chebyshev: {params.offset_chebyshev}",
        f"epsilon: {params.epsilon}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_trial(
    trial_idx: int,
    params: TrialParams,
    solver_bin: Path,
    problem_paths: Sequence[Path],
    args: argparse.Namespace,
    tmp_dir: Path,
) -> TrialResult:
    config_path = tmp_dir / f"heuristic_trial_{trial_idx:03d}.yaml"
    write_config(config_path, params)

    original_env = os.environ.get("HEURISTIC_CONFIG_PATH")
    os.environ["HEURISTIC_CONFIG_PATH"] = str(config_path)
    reload_heuristic_config()

    trial_metrics: List[Dict[str, Any]] = []
    solved = 0
    total = 0
    sum_h = 0.0
    sum_time = 0.0

    for problem_path in problem_paths:
        result = run_one_problem(
            solver=solver_bin,
            problem_path=problem_path,
            w_weight=args.w,
            time_limit_s=args.time_limit,
            fast_params=None,
            alpha=args.alpha,
            tie_break=args.tie_break,
            max_depth=args.max_depth,
            k_top_moves=args.k_top_moves,
        )
        trial_metrics.append({
            "problem": str(problem_path),
            "solved": result["solved"],
            "partial": result["partial"],
            "h_final": result["h_final"],
            "time_s": result["time_s"],
        })
        total += 1
        if result["solved"]:
            solved += 1
        sum_h += float(result["h_final"])
        sum_time += float(result["time_s"])

    avg_h = sum_h / total if total else float("inf")
    avg_time = sum_time / total if total else float("inf")

    if original_env is None:
        os.environ.pop("HEURISTIC_CONFIG_PATH", None)
    else:
        os.environ["HEURISTIC_CONFIG_PATH"] = original_env
    reload_heuristic_config()

    return TrialResult(
        index=trial_idx,
        params=params,
        solved=solved,
        total=total,
        avg_h=avg_h,
        avg_time=avg_time,
        metrics=trial_metrics,
    )


def suggest_ranges(results: List[TrialResult], top_k: int) -> Dict[str, Tuple[float, float]]:
    if not results:
        return {}
    ranked = sorted(
        results,
        key=lambda r: (-r.solved_rate, r.avg_h, r.avg_time, r.index),
    )
    window = ranked[: max(1, min(top_k, len(ranked)))]
    keys = [
        ("weight_euclidean", lambda p: p.weight_euclidean),
        ("weight_chebyshev", lambda p: p.weight_chebyshev),
        ("offset_euclidean", lambda p: p.offset_euclidean),
        ("offset_chebyshev", lambda p: p.offset_chebyshev),
    ]
    suggestions: Dict[str, Tuple[float, float]] = {}
    for name, getter in keys:
        values = [getter(r.params) for r in window]
        if not values:
            continue
        v_min = min(values)
        v_max = max(values)
        span = max(v_max - v_min, 1e-6)
        margin = 0.1 * span
        suggestions[name] = (max(0.0, v_min - margin), v_max + margin)
    return suggestions


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Sample heuristic blends, evaluate solver, and suggest refined ranges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--size", type=int, required=True, help="Board size to evaluate (e.g., 8)")
    ap.add_argument("--problems-root", type=Path, default=Path("problems"))
    ap.add_argument("--max-problems", type=int, default=10, help="Number of problems per trial")
    ap.add_argument("--trials", type=int, default=100, help="How many parameter sets to sample")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    ap.add_argument("--solver-bin", type=Path, default=Path("core/wastar"))
    ap.add_argument("--w", type=float, default=2.0, help="W weight for Weighted A*")
    ap.add_argument("--alpha", type=float, default=None, help="Optional ALPHA override")
    ap.add_argument("--tie-break", type=str, default=None, choices=["h_min", "h_max", "g_min", "g_max"], help="Tie-break mode")
    ap.add_argument("--time-limit", type=float, default=None, help="Per-problem time limit in seconds")
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--k-top-moves", type=int, default=None)
    ap.add_argument("--weight-euclid-range", type=float, nargs="*", default=[0.1, 1.5])
    ap.add_argument("--weight-cheby-range", type=float, nargs="*", default=[0.0, 1.5])
    ap.add_argument("--offset-euclid-range", type=float, nargs="*", default=[0.5, 2.5])
    ap.add_argument("--offset-cheby-range", type=float, nargs="*", default=[0.5, 2.5])
    ap.add_argument("--epsilon", type=float, default=1e-6)
    ap.add_argument("--normalize-weights", action="store_true", help="Normalize sampled weights to sum to 1")
    ap.add_argument("--top-k", type=int, default=10, help="How many top trials to use for range suggestion")
    ap.add_argument("--output", type=Path, default=Path("artifacts/heuristic_sweeps/results.json"))
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    args.weight_euclid_range = _parse_range(args.weight_euclid_range, "weight-euclid-range")
    args.weight_cheby_range = _parse_range(args.weight_cheby_range, "weight-cheby-range")
    args.offset_euclid_range = _parse_range(args.offset_euclid_range, "offset-euclid-range")
    args.offset_cheby_range = _parse_range(args.offset_cheby_range, "offset-cheby-range")

    rng = random.Random(args.seed)

    ensure_cli_solver(args.solver_bin)
    problems = list_problem_files(args.problems_root, args.size, args.max_problems)
    if not problems:
        raise RuntimeError("No problems found for the requested size")

    results: List[TrialResult] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="heuristic_sweep_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for trial_idx in range(args.trials):
            params = sample_params(rng, args)
            result = evaluate_trial(trial_idx, params, args.solver_bin, problems, args, tmp_dir)
            results.append(result)
            print(
                f"[trial {trial_idx+1:03d}/{args.trials}] rate={result.solved_rate:.3f} "
                f"avg_h={result.avg_h:.3f} avg_time={result.avg_time:.3f} "
                f"weights=({params.weight_euclidean:.3f}, {params.weight_chebyshev:.3f}) "
                f"offsets=({params.offset_euclidean:.3f}, {params.offset_chebyshev:.3f})",
                flush=True,
            )

    ranked = sorted(
        results,
        key=lambda r: (-r.solved_rate, r.avg_h, r.avg_time, r.index),
    )
    best = ranked[: min(5, len(ranked))]
    suggestions = suggest_ranges(results, args.top_k)

    payload = {
        "trials": [
            {
                "index": r.index,
                "params": {
                    "weight_euclidean": r.params.weight_euclidean,
                    "weight_chebyshev": r.params.weight_chebyshev,
                    "offset_euclidean": r.params.offset_euclidean,
                    "offset_chebyshev": r.params.offset_chebyshev,
                    "epsilon": r.params.epsilon,
                },
                "solved": r.solved,
                "total": r.total,
                "solved_rate": r.solved_rate,
                "avg_h": r.avg_h,
                "avg_time": r.avg_time,
                "metrics": r.metrics,
            }
            for r in ranked
        ],
        "suggestions": {
            k: {"min": v[0], "max": v[1]} for k, v in suggestions.items()
        },
        "summary": {
            "best_trial": best[0].index if best else None,
            "best_rate": best[0].solved_rate if best else None,
            "best_avg_h": best[0].avg_h if best else None,
            "best_avg_time": best[0].avg_time if best else None,
        },
    }

    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nTop trials:")
    for r in best:
        print(
            f"  trial {r.index:03d}: rate={r.solved_rate:.3f}, avg_h={r.avg_h:.3f}, "
            f"avg_time={r.avg_time:.3f}, weights=({r.params.weight_euclidean:.3f}, {r.params.weight_chebyshev:.3f}), "
            f"offsets=({r.params.offset_euclidean:.3f}, {r.params.offset_chebyshev:.3f})",
        )

    if suggestions:
        print("\nSuggested next ranges:")
        for name, (lo, hi) in suggestions.items():
            print(f"  {name}: [{lo:.4f}, {hi:.4f}]")
    else:
        print("\nNot enough data to suggest refined ranges.")


if __name__ == "__main__":
    main()
