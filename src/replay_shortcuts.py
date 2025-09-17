from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.io import load_problem, apply_ops
from utils.metrics import heuristic_value


Grid = List[List[int]]


def save_board(path: Path, size: int, grid: Grid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"size": size, "entities": grid}, f, ensure_ascii=False, indent=2)


def fingerprint(grid: Grid) -> str:
    # Non-linear 64-bit XOR-based fingerprint with rotation to avoid positional cancellation.
    # Keep in sync with rust/replay_shortcuts/src/main.rs
    MASK = (1 << 64) - 1
    def mix64(x: int) -> int:
        x &= MASK
        x ^= (x >> 30)
        x = (x * 0xBF58476D1CE4E5B9) & MASK
        x ^= (x >> 27)
        x = (x * 0x94D049BB133111EB) & MASK
        x ^= (x >> 31)
        return x & MASK

    acc = 0x9E3779B97F4A7C15  # seed
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            v64 = v & MASK  # two's complement view
            idx = ((r << 32) ^ c) & MASK
            k = (idx ^ 0xD6E8FEB86659FD93) & MASK
            x = mix64(v64 ^ k)
            acc = ((acc << 1) | (acc >> 63)) & MASK
            acc ^= x
    return f"{acc:016x}"


def gen_candidates(grid: Grid, n_max: int = 5) -> List[Tuple[int, int, int]]:
    """Generate rotation candidates similar to the C++ helper.

    Uses stride = max(1, n-1) to cover the board while limiting branching.
    Returns a list of (x, y, n).
    """
    size = len(grid)
    out: List[Tuple[int, int, int]] = []
    for n in range(2, min(size, n_max) + 1):
        # Use a coarser stride for n=2 as well to reduce branching
        step = max(2, n - 1)
        for y in range(0, size - n + 1, step):
            for x in range(0, size - n + 1, step):
                out.append((x, y, n))
    return out


def rotate_once_inplace(grid: Grid, x: int, y: int, n: int) -> None:
    # Apply one clockwise rotation of subgrid (y..y+n-1, x..x+n-1)
    sub = [[grid[y + i][x + j] for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            grid[y + i][x + j] = sub[n - 1 - j][i]


def best_candidates_by_heuristic(grid: Grid, cand: List[Tuple[int, int, int]], top_k: int) -> List[Tuple[int, int, int]]:
    """Return up to top_k candidates that reduce heuristic (or keep equal), sorted by new h."""
    h0 = heuristic_value(grid)
    scored: List[Tuple[int, Tuple[int, int, int]]] = []
    size = len(grid)
    # Work on a local copy for transforms
    for (x, y, n) in cand:
        # bounds are guaranteed by gen_candidates
        g2 = [row[:] for row in grid]
        rotate_once_inplace(g2, x, y, n)
        h1 = heuristic_value(g2)
        if h1 <= h0:
            scored.append((h1, (x, y, n)))
    scored.sort(key=lambda t: t[0])
    return [op for _, op in scored[:top_k]]


def bfs_shortcut(
    start_grid: Grid,
    start_idx: int,
    known_index: Dict[str, int],
    max_depth: int = 2,
    n_max: int = 5,
    top_k: int = 12,
    max_nodes: int = 2000,
) -> Optional[Dict[str, Any]]:
    """Bounded BFS guided by heuristic to find a jump to a later known state.

    Returns a dict: {start, target, depth, improvement, ops} or None if none found.
    """
    start_fp = fingerprint(start_grid)
    # Ignore if start itself isn't known (should be), but continue anyway
    best: Optional[Dict[str, Any]] = None
    visited = {start_fp}
    Q = deque()
    # Initialize frontier with best single-step candidates
    base_cand = gen_candidates(start_grid, n_max=n_max)
    base_best = best_candidates_by_heuristic(start_grid, base_cand, top_k)
    for (x, y, n) in base_best:
        g1 = [row[:] for row in start_grid]
        rotate_once_inplace(g1, x, y, n)
        fp1 = fingerprint(g1)
        if fp1 in visited:
            continue
        visited.add(fp1)
        Q.append((g1, [(x, y, n)], 1))

    nodes = 0
    while Q:
        g, ops, d = Q.popleft()
        nodes += 1
        if nodes > max_nodes:
            break

        fp = fingerprint(g)
        if fp in known_index:
            j = known_index[fp]
            # Only interesting if it lands in a strictly later step than consuming d moves
            if j > start_idx + d:
                improvement = j - (start_idx + d)
                cur = {
                    "start": start_idx,
                    "target": j,
                    "depth": d,
                    "improvement": improvement,
                    "ops": [
                        {"x": x, "y": y, "n": n} for (x, y, n) in ops
                    ],
                }
                if best is None or (improvement > best["improvement"]) or (
                    improvement == best["improvement"] and d < best["depth"]
                ):
                    best = cur
                # Early exit if a decent jump found
                if improvement >= 5:
                    return best

        if d >= max_depth:
            continue

        # Expand guided by heuristic
        cand = gen_candidates(g, n_max=n_max)
        picks = best_candidates_by_heuristic(g, cand, top_k)
        for (x, y, n) in picks:
            g2 = [row[:] for row in g]
            rotate_once_inplace(g2, x, y, n)
            fp2 = fingerprint(g2)
            if fp2 in visited:
                continue
            visited.add(fp2)
            Q.append((g2, ops + [(x, y, n)], d + 1))

    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay ops, save each board, and search shortcuts to known states")
    ap.add_argument("--problem", type=str, default="test.json", help="path to problem JSON")
    ap.add_argument("--ops", type=str, default="test.ops.json", help="path to ops JSON {ops: [...]} or list")
    ap.add_argument("--known_ops", type=str, default=None, help="optional path to another ops JSON to include as known states (e.g., test.solution.json)")
    ap.add_argument("--out_dir", type=str, default="artifacts/replay_test", help="output directory for states and results")
    ap.add_argument("--save_states", action="store_true", help="save each intermediate board state to out_dir/states/")
    ap.add_argument("--search_depth", type=int, default=2)
    ap.add_argument("--n_max", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--max_nodes", type=int, default=2000)
    ap.add_argument("--max_starts", type=int, default=None, help="limit number of starting stages to search from")
    args = ap.parse_args()

    prob = load_problem(args.problem)
    size = int(prob["size"])
    grid0: Grid = prob["entities"]

    # Load ops (support two formats: {"ops": [...]} or list)
    with open(args.ops, "r", encoding="utf-8") as f:
        data_ops = json.load(f)
    if isinstance(data_ops, dict) and "ops" in data_ops:
        ops_main: List[Dict[str, int]] = list(data_ops["ops"])  # type: ignore
    elif isinstance(data_ops, list):
        ops_main = list(data_ops)  # type: ignore
    else:
        raise ValueError("Invalid ops format: expected {\"ops\": [...]} or a list")

    # Replay and collect known states from main ops
    out_dir = Path(args.out_dir)
    states_dir = out_dir / "states"
    states: List[Grid] = [grid0]
    if args.save_states:
        save_board(states_dir / f"step_{0:04d}.json", size, grid0)
    cur = [row[:] for row in grid0]
    for i, op in enumerate(ops_main, start=1):
        cur = apply_ops(cur, [op])
        states.append(cur)
        if args.save_states:
            save_board(states_dir / f"step_{i:04d}.json", size, cur)

    # Optionally include known states from another ops list (e.g., solution)
    known_map: Dict[str, int] = {}
    for idx, g in enumerate(states):
        known_map[fingerprint(g)] = idx

    if args.known_ops:
        with open(args.known_ops, "r", encoding="utf-8") as f:
            known_data = json.load(f)
        if isinstance(known_data, dict) and "ops" in known_data:
            ops_known = list(known_data["ops"])  # type: ignore
        elif isinstance(known_data, list):
            ops_known = list(known_data)  # type: ignore
        else:
            raise ValueError("Invalid known_ops format: expected {\"ops\": [...]} or a list")
        gk = [row[:] for row in grid0]
        # include intermediate states from known path as well, but map to their step idx beyond main if not present
        for j, op in enumerate(ops_known, start=1):
            gk = apply_ops(gk, [op])
            fp = fingerprint(gk)
            if fp not in known_map:
                # use a synthetic large index to denote later-than-main states
                known_map[fp] = len(states) + j

    # Search shortcuts from each stage
    shortcuts: List[Dict[str, Any]] = []
    max_starts = args.max_starts if args.max_starts is not None else len(states) - 1
    for i in range(min(max_starts, len(states) - 1)):
        res = bfs_shortcut(
            start_grid=states[i],
            start_idx=i,
            known_index=known_map,
            max_depth=args.search_depth,
            n_max=args.n_max,
            top_k=args.top_k,
            max_nodes=args.max_nodes,
        )
        if res:
            shortcuts.append(res)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "shortcuts.json").open("w", encoding="utf-8") as f:
        json.dump({
            "problem": str(args.problem),
            "ops": str(args.ops),
            "known_ops": str(args.known_ops) if args.known_ops else None,
            "search_depth": args.search_depth,
            "n_max": args.n_max,
            "top_k": args.top_k,
            "max_nodes": args.max_nodes,
            "starts_considered": int(min(max_starts, len(states) - 1)),
            "shortcuts_found": shortcuts,
        }, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "saved_states": bool(args.save_states),
        "states_count": len(states),
        "starts_considered": int(min(max_starts, len(states) - 1)),
        "shortcuts_found": len(shortcuts),
        "out_dir": str(out_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
