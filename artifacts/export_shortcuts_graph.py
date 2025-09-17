#!/usr/bin/env python3
from __future__ import annotations

import json
import glob
from pathlib import Path


def collect_shortcuts(patterns: list[str]) -> list[dict]:
    items: list[dict] = []
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                d = json.loads(Path(p).read_text())
            except Exception:
                continue
            depth = int(d.get("search_depth", -1))
            for sc in d.get("shortcuts_found", []):
                items.append(
                    {
                        "start": int(sc["start"]),
                        "target": int(sc["target"]),
                        "depth": int(sc["depth"]),
                        "improvement": int(sc["improvement"]),
                        "ops": sc.get("ops", []),
                        "run_depth": depth,
                        "source": str(p),
                    }
                )
    return items


def build_graph(shortcuts: list[dict]) -> dict:
    # Nodes: unique step indices that appear in any shortcut (start or target)
    steps = sorted(
        set([sc["start"] for sc in shortcuts]) | set([sc["target"] for sc in shortcuts])
    )
    index = {s: i for i, s in enumerate(steps)}
    nodes = [{"id": i, "step": s} for s, i in index.items()]
    links = []
    for sc in shortcuts:
        links.append(
            {
                "source": index[sc["start"]],
                "target": index[sc["target"]],
                "improvement": sc["improvement"],
                "depth": sc["depth"],
                "run_depth": sc["run_depth"],
                "source_file": sc["source"],
            }
        )
    return {"nodes": nodes, "links": links}


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    pats = [
        "artifacts/replay_bench_d1_scan_*/shortcuts.json",
        "artifacts/replay_bench_d2_scan_*/shortcuts.json",
        "artifacts/replay_bench_d3_scan_*/shortcuts.json",
        "artifacts/replay_bench_d4_scan_*/shortcuts.json",
    ]
    items = collect_shortcuts(pats)
    items.sort(key=lambda x: (x["start"], x["target"]))
    graph = build_graph(items)
    Path("artifacts/shortcuts_graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "shortcuts": len(items),
                "nodes": len(graph["nodes"]),
                "links": len(graph["links"]),
                "out": "artifacts/shortcuts_graph.json",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

