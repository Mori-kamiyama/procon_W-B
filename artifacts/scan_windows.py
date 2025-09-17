#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Tuple


def run_bin(bin_path: Path, problem: Path, ops: Path, out_dir: Path,
            depth: int, n_max: int, top_k: int, max_nodes: int,
            start_begin: int, start_end: int,
            focus_off: bool, lds_limit: int,
            stride: int | None = None,
            no_h_gate: bool = False,
            stochastic: bool = False,
            prefix_depth: int = 3,
            walks: int = 512) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(bin_path),
        "--problem", str(problem),
        "--ops", str(ops),
        "--out-dir", str(out_dir),
        "--search-depth", str(depth),
        "--n-max", str(n_max),
        "--top-k", str(top_k),
        "--max-nodes", str(max_nodes),
        "--start-begin", str(start_begin),
        "--start-end", str(start_end),
        "--lds-limit", str(lds_limit),
    ]
    if focus_off:
        cmd.append("--focus-off")
    if stride is not None:
        cmd += ["--stride", str(stride)]
    if no_h_gate:
        cmd.append("--no-h-gate")
    if stochastic:
        cmd += ["--stochastic", "--prefix-depth", str(prefix_depth), "--walks", str(walks)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # binary prints a JSON one-liner summary to stdout
    summary = json.loads(proc.stdout.strip())
    # load shortcuts file for details
    sc_path = out_dir / "shortcuts.json"
    details = {}
    if sc_path.exists():
        details = json.loads(sc_path.read_text())
    return {"summary": summary, "details": details}


def summarize(details: dict) -> Tuple[int, int, int, List[Tuple[int, int, int]]]:
    sc = details.get("shortcuts_found", [])
    imps = [int(s.get("improvement", 0)) for s in sc]
    best = max(imps) if imps else 0
    return len(sc), sum(imps), best, [(s["start"], s["target"], s["improvement"]) for s in sc[:5]]


def main():
    ap = argparse.ArgumentParser(description="Scan windows for shortcuts via the Rust binary")
    ap.add_argument("--bin", type=Path, default=Path("rust/replay_shortcuts/target/release/replay_shortcuts"))
    ap.add_argument("--problem", type=Path, default=Path("test.json"))
    ap.add_argument("--ops", type=Path, default=Path("test.ops.json"))
    ap.add_argument("--base_out", type=Path, default=Path("artifacts"))
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--windows", type=str, nargs="+", help="List like 300:340 340:380", required=True)
    ap.add_argument("--n_max", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--max_nodes", type=int, default=50000)
    ap.add_argument("--lds_limit", type=int, default=3)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--no_h_gate", action="store_true")
    ap.add_argument("--focus_off", action="store_true")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--prefix_depth", type=int, default=3)
    ap.add_argument("--walks", type=int, default=512)
    args = ap.parse_args()

    out_rows = []
    for w in args.windows:
        try:
            b_s, e_s = w.split(":", 1)
            b, e = int(b_s), int(e_s)
        except Exception:
            print(f"skip invalid window spec: {w}")
            continue
        out_dir = args.base_out / f"replay_bench_d{args.depth}_scan_w{b}_{e}"
        res = run_bin(
            args.bin, args.problem, args.ops, out_dir,
            args.depth, args.n_max, args.top_k, args.max_nodes,
            b, e, args.focus_off, args.lds_limit,
            stride=args.stride, no_h_gate=args.no_h_gate,
            stochastic=args.stochastic, prefix_depth=args.prefix_depth, walks=args.walks,
        )
        cnt, tot, best, first = summarize(res.get("details", {}))
        out_rows.append((w, cnt, tot, best))
        print(f"d{args.depth} {w}: count={cnt}, total={tot}, best={best}")
    # Write csv summary
    if out_rows:
        csv_path = args.base_out / f"scan_d{args.depth}_summary.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("window,count,total,best\n")
            for w, c, t, b in out_rows:
                f.write(f"{w},{c},{t},{b}\n")
        print(f"Summary written: {csv_path}")


if __name__ == "__main__":
    main()
