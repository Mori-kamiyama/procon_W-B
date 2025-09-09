from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

Grid = List[List[int]]


def pair_positions(grid: Grid) -> Dict[int, List[Tuple[int, int]]]:
    """Return mapping value -> list of coordinates where it appears.

    Many puzzles here have exactly two occurrences for each value, but we do not
    assume; we simply collect all.
    """
    pos: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            pos[v].append((r, c))
    return pos


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristic_value(grid: Grid) -> int:
    """Heuristic used by old solvers: sum(max(0, dist - 1)) over all pairs.

    If a value appears more than twice, we consider the best two occurrences.
    If it appears once or zero times, contributes 0.
    """
    pos = pair_positions(grid)
    total = 0
    for v, coords in pos.items():
        if len(coords) < 2:
            continue
        # use two farthest or first two? The C++ solver pairs the two locations in input.
        # Here we use the first two for consistency with typical two-occurrence datasets.
        a, b = coords[0], coords[1]
        dist = manhattan(a, b)
        if dist > 1:
            total += dist - 1
    return int(total)


def pairs_and_total(grid: Grid) -> Tuple[int, int]:
    """Return (#adjacent pairs, total #pairs) based on Manhattan distance == 1.

    If a value appears two or more times, we count exactly one pair via the first
    two occurrences. This matches the heuristic’s pairing assumption.
    """
    pos = pair_positions(grid)
    total_pairs = 0
    adjacent_pairs = 0
    for v, coords in pos.items():
        if len(coords) >= 2:
            total_pairs += 1
            a, b = coords[0], coords[1]
            if manhattan(a, b) == 1:
                adjacent_pairs += 1
    return adjacent_pairs, total_pairs


def pairs_rate(grid: Grid) -> float:
    adj, tot = pairs_and_total(grid)
    if tot == 0:
        return 1.0
    return adj / float(tot)
