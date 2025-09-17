"""Generate teacher data by solving puzzles with the weighted A* solver.

This script runs the existing C++ solver on a batch of problems, replays the
returned operation sequences, and emits training samples of the form
"(board state before move, teacher move)".  The generated file is a JSON Lines
(`.jsonl`) file so that downstream training can be streamed efficiently.

Example usage (generate 3k samples for 6x6 problems)::

    python3 src/generate_teacher_data.py \
        --size 6 \
        --target_samples 3000 \
        --output artifacts/datasets/teacher_greedy_6x6.jsonl

The command will stop once the requested number of samples has been gathered or
raise an error if the available problems and solutions do not provide enough
steps.
"""

from __future__ import annotations

import argparse
import json
import sys
import itertools
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from run_search import ensure_cli_solver, run_one_problem
from utils.io import list_problem_files, load_problem
from utils.metrics import heuristic_value, pairs_rate


Grid = List[List[int]]
Op = Dict[str, int]


def _clone_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def _rotate_once_inplace(grid: Grid, x: int, y: int, n: int) -> None:
    """Apply one clockwise rotation to the (y:y+n, x:x+n) sub-grid."""

    sub = [[grid[y + i][x + j] for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            grid[y + i][x + j] = sub[n - 1 - j][i]


def _apply_single_op(grid: Grid, op: Op) -> Grid:
    g2 = _clone_grid(grid)
    _rotate_once_inplace(g2, int(op["x"]), int(op["y"]), int(op["n"]))
    return g2


def iter_samples(
    problem_path: Path,
    ops: Sequence[Op],
    *,
    extra_metadata: Dict[str, object] | None = None,
) -> Iterable[Dict[str, object]]:
    """Yield training samples for each step of the provided solution."""

    data = load_problem(problem_path)
    grid = data["entities"]

    current = _clone_grid(grid)
    for step_idx, op in enumerate(ops):
        before = _clone_grid(current)
        h_before = heuristic_value(before)
        pr_before = pairs_rate(before)

        after = _apply_single_op(before, op)
        h_after = heuristic_value(after)
        pr_after = pairs_rate(after)

        sample: Dict[str, object] = {
            "problem": str(problem_path),
            "step_index": step_idx,
            "grid": before,
            "move": {"x": int(op["x"]), "y": int(op["y"]), "n": int(op["n"])},
            "h_before": h_before,
            "h_after": h_after,
            "pairs_rate_before": pr_before,
            "pairs_rate_after": pr_after,
        }
        if extra_metadata:
            sample.update(extra_metadata)
        yield sample

        current = after


def generate_teacher_data(
    *,
    size: int,
    target_samples: int,
    problems_root: Path,
    solver_bin: Path,
    solver_src: Path | None,
    w_weight: float,
    time_limit_s: float | None,
    fast_max_small_n: int | None,
    fast_cand_cap: int | None,
    fast_topk: int | None,
    alpha: float | None,
    tie_break: str | None,
    max_depth: int | None,
    k_top_moves: int | None,
    output_path: Path,
    scramble_on_shortfall: bool,
    scramble_min: int,
    scramble_max: int,
    scramble_seed: int,
) -> int:
    """Run solver on problems and dump samples to output_path.

    Returns the total number of samples written.
    """

    ensure_cli_solver(solver_bin, solver_src)

    files = list_problem_files(problems_root, size)
    if not files:
        raise RuntimeError(f"No problem files found for size {size} under {problems_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    fast_params = {}
    if fast_max_small_n is not None:
        fast_params["FAST_MAX_SMALL_N"] = fast_max_small_n
    if fast_cand_cap is not None:
        fast_params["FAST_CAND_CAP"] = fast_cand_cap
    if fast_topk is not None:
        fast_params["FAST_TOPK"] = fast_topk
    fast_params = fast_params or None

    with output_path.open("w", encoding="utf-8") as fout:
        for problem_path in files:
            if written >= target_samples:
                break

            result = run_one_problem(
                solver=solver_bin,
                problem_path=problem_path,
                w_weight=w_weight,
                time_limit_s=time_limit_s,
                fast_params=fast_params,
                alpha=alpha,
                tie_break=tie_break,
                max_depth=max_depth,
                k_top_moves=k_top_moves,
            )

            if not result["ok"]:
                print(
                    f"[warn] solver did not produce a valid solution for {problem_path}: {result['error']}",
                    file=sys.stderr,
                )
                continue

            ops: Sequence[Op] = result["ops"]
            metadata = {"source": {"type": "original", "problem": str(problem_path)}}
            for sample in iter_samples(problem_path, ops, extra_metadata=metadata):
                json.dump(sample, fout, ensure_ascii=False)
                fout.write("\n")
                written += 1
                if written >= target_samples:
                    break

        if written < target_samples and scramble_on_shortfall:
            rng = random.Random(scramble_seed)
            if scramble_min < 1 or scramble_max < scramble_min:
                raise ValueError("Invalid scramble range")
            generated_dir = output_path.parent / f"{output_path.stem}_problems"
            generated_dir.mkdir(parents=True, exist_ok=True)

            scramble_index = 0
            for base_problem in itertools.cycle(files):
                if written >= target_samples:
                    break

                base_data = load_problem(base_problem)
                size_local = int(base_data["size"])
                scramble_depth = rng.randint(scramble_min, scramble_max)
                scrambled = _clone_grid(base_data["entities"])
                for _ in range(scramble_depth):
                    n = rng.randint(2, size_local)
                    x = rng.randint(0, size_local - n)
                    y = rng.randint(0, size_local - n)
                    _rotate_once_inplace(scrambled, x, y, n)

                tmp_path = generated_dir / f"scramble_{scramble_index:05d}.json"
                with tmp_path.open("w", encoding="utf-8") as f:
                    json.dump({"size": size_local, "entities": scrambled}, f, ensure_ascii=False)

                result = run_one_problem(
                    solver=solver_bin,
                    problem_path=tmp_path,
                    w_weight=w_weight,
                    time_limit_s=time_limit_s,
                    fast_params=fast_params,
                    alpha=alpha,
                    tie_break=tie_break,
                    max_depth=max_depth,
                    k_top_moves=k_top_moves,
                )

                if not result["ok"]:
                    print(
                        f"[warn] solver failed on scrambled problem {tmp_path}: {result['error']}",
                        file=sys.stderr,
                    )
                    scramble_index += 1
                    continue

                metadata = {
                    "source": {
                        "type": "scramble",
                        "base_problem": str(base_problem),
                        "scramble_depth": scramble_depth,
                        "scramble_index": scramble_index,
                        "scramble_seed": scramble_seed,
                    }
                }
                ops = result["ops"]
                for sample in iter_samples(tmp_path, ops, extra_metadata=metadata):
                    json.dump(sample, fout, ensure_ascii=False)
                    fout.write("\n")
                    written += 1
                    if written >= target_samples:
                        break

                scramble_index += 1

    if written < target_samples:
        raise RuntimeError(
            f"Insufficient samples: requested {target_samples} but only {written} were generated"
        )

    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate teacher data using greedy solver runs")
    ap.add_argument("--size", type=int, default=6, help="board size (e.g., 6)")
    ap.add_argument("--target_samples", type=int, default=3000, help="number of samples to generate")
    ap.add_argument("--problems_root", type=str, default="problems", help="root directory of problems")
    ap.add_argument("--solver_bin", type=str, default="core/wastar", help="path to solver binary")
    ap.add_argument("--solver_src", type=str, default=None, help="optional path to solver source")
    ap.add_argument("--w", type=float, default=2.0, help="weight for weighted A*")
    ap.add_argument("--time_limit_s", type=float, default=None, help="time limit per problem in seconds")
    ap.add_argument("--fast_max_small_n", type=int, default=None)
    ap.add_argument("--fast_cand_cap", type=int, default=None)
    ap.add_argument("--fast_topk", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None, help="alpha parameter for composite heuristic")
    ap.add_argument("--tie_break", type=str, default=None, choices=["h_min", "h_max", "g_min", "g_max"], help="tie-break rule")
    ap.add_argument("--max_depth", type=int, default=None, help="max search depth")
    ap.add_argument("--k_top_moves", type=int, default=None, help="alias for FAST_TOPK")
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="output JSONL path (default: artifacts/datasets/teacher_greedy_{size}x{size}.jsonl)",
    )
    ap.add_argument(
        "--no-scramble",
        dest="scramble_on_shortfall",
        action="store_false",
        help="disable random scrambling to top up samples when originals are insufficient",
    )
    ap.add_argument("--scramble-min", type=int, default=3, help="minimum scramble operations when augmenting")
    ap.add_argument("--scramble-max", type=int, default=8, help="maximum scramble operations when augmenting")
    ap.add_argument("--scramble-seed", type=int, default=0, help="random seed for scramble augmentation")
    ap.set_defaults(scramble_on_shortfall=True)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    size = int(args.size)
    if args.output is None:
        output_path = Path("artifacts/datasets") / f"teacher_greedy_{size}x{size}.jsonl"
    else:
        output_path = Path(args.output)

    written = generate_teacher_data(
        size=size,
        target_samples=int(args.target_samples),
        problems_root=Path(args.problems_root),
        solver_bin=Path(args.solver_bin),
        solver_src=Path(args.solver_src) if args.solver_src else None,
        w_weight=float(args.w),
        time_limit_s=float(args.time_limit_s) if args.time_limit_s is not None else None,
        fast_max_small_n=args.fast_max_small_n,
        fast_cand_cap=args.fast_cand_cap,
        fast_topk=args.fast_topk,
        alpha=args.alpha,
        tie_break=args.tie_break,
        max_depth=args.max_depth,
        k_top_moves=args.k_top_moves,
        output_path=output_path,
        scramble_on_shortfall=bool(args.scramble_on_shortfall),
        scramble_min=int(args.scramble_min),
        scramble_max=int(args.scramble_max),
        scramble_seed=int(args.scramble_seed),
    )

    print(json.dumps({
        "size": size,
        "target_samples": int(args.target_samples),
        "samples_written": written,
        "output": str(output_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

