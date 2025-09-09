from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

Grid = List[List[int]]


def load_problem(path: str | Path) -> Dict[str, Any]:
    """Load a single problem JSON file.

    Expected format:
    {
      "size": int,
      "entities": [[int, ...], ...]
    }
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Basic validation
    size = int(data["size"])  # may raise KeyError, ValueError
    entities = data["entities"]
    if not isinstance(entities, list) or len(entities) != size:
        raise ValueError(f"Invalid entities shape in {p}")
    for row in entities:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(f"Invalid row length in {p}")
    return {"size": size, "entities": entities}


def list_problem_files(root: str | Path, size: int, limit: int | None = None) -> List[Path]:
    """List problem files under problems/{size}x{size}.

    Returns a sorted list of Paths. If limit is provided, returns up to that many.
    """
    root_path = Path(root)
    dir_path = root_path / f"{size}x{size}"
    if not dir_path.exists():
        raise FileNotFoundError(f"Problem directory not found: {dir_path}")
    files = sorted([p for p in dir_path.iterdir() if p.suffix == ".json"])
    if limit is not None:
        files = files[: int(limit)]
    return files


def apply_ops(grid: Grid, ops: List[Dict[str, int]]) -> Grid:
    """Apply rotation operations to a grid and return a new grid.

    Operation format: {"x": int, "y": int, "n": int} meaning rotate the n x n
    subgrid at top-left (y, x) clockwise once.
    """
    size = len(grid)
    g = [row[:] for row in grid]

    def rotate_once(x: int, y: int, n: int) -> None:
        sub = [[g[y + i][x + j] for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                g[y + i][x + j] = sub[n - 1 - j][i]

    for op in ops:
        x, y, n = int(op["x"]), int(op["y"]), int(op["n"])
        if not (0 <= x < size and 0 <= y < size and 2 <= n <= size and x + n <= size and y + n <= size):
            raise ValueError(f"Invalid operation {op} for size {size}")
        rotate_once(x, y, n)
    return g


def save_solution(path: str | Path, ops: List[Dict[str, int]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump({"ops": ops}, f, ensure_ascii=False, indent=2)
