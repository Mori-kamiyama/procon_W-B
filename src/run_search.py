from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.io import apply_ops, list_problem_files, load_problem, save_solution
from utils.metrics import heuristic_value, pairs_rate


def ensure_cli_solver(binary_path: Path, source_cpp: Optional[Path] = None) -> None:
    """Ensure a CLI solver binary exists.

    If not found, attempt to compile using g++ (or clang++ fallback). By default,
    compile the solver from core/wastar.cpp into `binary_path`.
    """
    if source_cpp is None:
        source_cpp = Path("core/wastar.cpp").resolve()
    # Rebuild if missing or source is newer
    if binary_path.exists() and os.access(binary_path, os.X_OK):
        try:
            if binary_path.stat().st_mtime >= source_cpp.stat().st_mtime:
                return
        except FileNotFoundError:
            pass
    if not source_cpp.exists():
        # If binary already exists, keep using it without rebuilding
        if binary_path.exists() and os.access(binary_path, os.X_OK):
            return
        raise FileNotFoundError(f"Solver source not found: {source_cpp}")

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    cxx = shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        raise RuntimeError("No C++ compiler found (g++ or clang++). Please install one.")

    cmd = [
        cxx,
        "-O3",
        "-std=c++20",
        str(source_cpp),
        "-o",
        str(binary_path),
    ]
    print(f"[build] Compiling solver → {binary_path}")
    subprocess.run(cmd, check=True)


def run_one_problem(
    solver: Path,
    problem_path: Path,
    w_weight: float = 2.0,
    time_limit_s: Optional[float] = None,
    fast_params: Optional[Dict[str, int]] = None,
    alpha: Optional[float] = None,
    tie_break: Optional[str] = None,
    max_depth: Optional[int] = None,
    k_top_moves: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the CLI solver on one problem and return metrics and result.

    Returns a dict with fields:
    - ok: bool
    - timeout: bool
    - time_s: float
    - ops: List[{x,y,n}] (may be empty)
    - pairs_rate: float (final grid)
    - h_final: int
    - ops_count: int
    - error: Optional[str]
    """
    data = load_problem(problem_path)
    grid0 = data["entities"]
    env = os.environ.copy()
    env["WASTAR_W"] = str(max(1.0, float(w_weight)))
    if alpha is not None:
        env["WASTAR_ALPHA"] = str(float(alpha))
    if tie_break is not None:
        env["TIE_BREAK"] = str(tie_break)
    if max_depth is not None:
        env["MAX_DEPTH"] = str(int(max_depth))
    if k_top_moves is not None:
        env["K_TOP_MOVES"] = str(int(k_top_moves))
    if fast_params:
        if "FAST_MAX_SMALL_N" in fast_params:
            env["FAST_MAX_SMALL_N"] = str(int(fast_params["FAST_MAX_SMALL_N"]))
        if "FAST_CAND_CAP" in fast_params:
            env["FAST_CAND_CAP"] = str(int(fast_params["FAST_CAND_CAP"]))
        if "FAST_TOPK" in fast_params:
            env["FAST_TOPK"] = str(int(fast_params["FAST_TOPK"]))
    if time_limit_s is not None and time_limit_s > 0:
        env["WASTAR_TIME_LIMIT_S"] = str(float(time_limit_s))
        env["TIME_LIMIT_S"] = str(float(time_limit_s))

    t0 = time.perf_counter()
    ops: List[Dict[str, int]] = []
    timeout = False
    err: Optional[str] = None
    try:
        cp = subprocess.run(
            [str(solver), str(problem_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=None,
            check=False,
        )
        if cp.returncode == 0:
            try:
                result = json.loads(cp.stdout)
                ops = list(result.get("ops", []))
            except json.JSONDecodeError as e:
                err = f"Invalid solver output JSON: {e}"
        else:
            err = cp.stderr.strip() or f"solver exited with code {cp.returncode}"
    except subprocess.TimeoutExpired:
        timeout = True
        err = "timeout"
    t1 = time.perf_counter()

    # Post metrics
    grid_final = apply_ops(grid0, ops) if ops else grid0
    pr = pairs_rate(grid_final)
    hfin = heuristic_value(grid_final)

    # Extract solver metrics if present
    solved = bool(result.get("solved", False)) if 'result' in locals() else False
    partial = bool(result.get("partial", False)) if 'result' in locals() else False
    metrics = result.get("metrics", {}) if 'result' in locals() else {}
    time_solver = metrics.get("time_s")
    nodes_expanded = metrics.get("nodes_expanded")
    nodes_generated = metrics.get("nodes_generated")
    open_max = metrics.get("open_max")
    peak_rss_mb = metrics.get("peak_rss_mb")

    return {
        "ok": err is None and not timeout and len(ops) > 0 and (solved or not partial),
        "timeout": timeout,
        "time_s": float(time_solver) if time_solver is not None else (t1 - t0),
        "ops": ops,
        "pairs_rate": float(pr),
        "h_final": int(hfin),
        "ops_count": int(len(ops)),
        "solved": solved,
        "partial": partial,
        "nodes_expanded": nodes_expanded,
        "nodes_generated": nodes_generated,
        "open_max": open_max,
        "peak_rss_mb": peak_rss_mb,
        "error": err,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run weighted A* search experiments")
    ap.add_argument("--size", type=int, required=True, help="board size (e.g., 6)")
    ap.add_argument("--problems_root", type=str, default="problems", help="problems root directory")
    ap.add_argument("--max_problems", type=int, default=5, help="number of problems to evaluate")
    ap.add_argument("--w", type=float, default=2.0, help="w weight for wA*")
    ap.add_argument("--time_limit_s", type=float, default=None, help="per-problem time limit (seconds)")
    ap.add_argument("--solver_bin", type=str, default="core/wastar", help="path to compiled solver binary")
    ap.add_argument("--save_dir", type=str, default="artifacts/results", help="where to save solutions")
    ap.add_argument("--fast_max_small_n", type=int, default=None)
    ap.add_argument("--fast_cand_cap", type=int, default=None)
    ap.add_argument("--fast_topk", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None, help="alpha for composite heuristic [0,1]")
    ap.add_argument("--tie_break", type=str, default=None, choices=["h_min","h_max","g_min","g_max"], help="tie-break rule")
    ap.add_argument("--max_depth", type=int, default=None, help="max search depth (g)")
    ap.add_argument("--k_top_moves", type=int, default=None, help="cap of best successors to keep (alias of FAST_TOPK)")
    args = ap.parse_args()

    solver_bin = Path(args.solver_bin)
    ensure_cli_solver(solver_bin)

    fast_params = {}
    if args.fast_max_small_n is not None:
        fast_params["FAST_MAX_SMALL_N"] = args.fast_max_small_n
    if args.fast_cand_cap is not None:
        fast_params["FAST_CAND_CAP"] = args.fast_cand_cap
    if args.fast_topk is not None:
        fast_params["FAST_TOPK"] = args.fast_topk

    files = list_problem_files(args.problems_root, args.size, args.max_problems)
    save_root = Path(args.save_dir) / f"{args.size}x{args.size}"
    save_root.mkdir(parents=True, exist_ok=True)

    times: List[float] = []
    prs: List[float] = []
    opcounts: List[int] = []
    open_max_list: List[int] = []
    nodes_expanded_list: List[int] = []
    nodes_generated_list: List[int] = []
    ok_count = 0

    for f in files:
        res = run_one_problem(
            solver=solver_bin,
            problem_path=f,
            w_weight=args.w,
            time_limit_s=args.time_limit_s,
            fast_params=fast_params or None,
            alpha=args.alpha,
            tie_break=args.tie_break,
            max_depth=args.max_depth,
            k_top_moves=args.k_top_moves,
        )
        times.append(res["time_s"])
        prs.append(res["pairs_rate"])
        opcounts.append(res["ops_count"])
        if res["ok"]:
            ok_count += 1
        if res.get("open_max") is not None:
            open_max_list.append(int(res["open_max"]))
        if res.get("nodes_expanded") is not None:
            nodes_expanded_list.append(int(res["nodes_expanded"]))
        if res.get("nodes_generated") is not None:
            nodes_generated_list.append(int(res["nodes_generated"]))
        # Save solution JSON regardless; empty ops means no solution found
        out_path = save_root / (f.stem + ".solution.json")
        save_solution(out_path, res["ops"])  # minimal artifact
        print(json.dumps({
            "problem": str(f),
            "ok": res["ok"],
            "timeout": res["timeout"],
            "time_s": res["time_s"],
            "pairs_rate": res["pairs_rate"],
            "ops_count": res["ops_count"],
            "open_max": res.get("open_max"),
            "nodes_expanded": res.get("nodes_expanded"),
            "nodes_generated": res.get("nodes_generated"),
            "error": res["error"],
        }, ensure_ascii=False))

    # Summary
    import statistics as stats
    summary = {
        "problems_evaluated": len(files),
        "ok_count": ok_count,
        "pairs_rate_avg": float(stats.mean(prs)) if prs else 0.0,
        "ops_avg": float(stats.mean([x for x in opcounts if x is not None])) if opcounts else 0.0,
        "time_avg_s": float(stats.mean(times)) if times else 0.0,
        "open_max_max": int(max(open_max_list)) if open_max_list else 0,
        "nodes_expanded_avg": float(stats.mean(nodes_expanded_list)) if nodes_expanded_list else None,
        "nodes_generated_avg": float(stats.mean(nodes_generated_list)) if nodes_generated_list else None,
    }
    print("SUMMARY:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
