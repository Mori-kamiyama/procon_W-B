from __future__ import annotations

from collections import defaultdict
import math
from typing import Dict, List, Optional, Tuple

from .heuristic_config import HeuristicConfig, get_heuristic_config

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


def heuristic_value(grid: Grid, config: Optional[HeuristicConfig] = None) -> float:
    """Return heuristic defined by Euclidean/Chebyshev blend from `heuristic.yaml`.

    For each value, we consider the first two occurrences (consistent with solver
    pairing). The contribution is

        w_euclid * max(0, ||diff||_2 - offset_euclid)
      + w_cheby  * max(0, ||diff||_inf - offset_chebyshev)

    Values appearing fewer than twice contribute 0.
    """
    cfg = config or get_heuristic_config()
    pos = pair_positions(grid)
    total = 0.0
    for coords in pos.values():
        if len(coords) < 2:
            continue
        a, b = coords[0], coords[1]
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        euclid = math.hypot(dr, dc)
        cheby = float(max(dr, dc))
        contrib = 0.0
        if cfg.weight_euclidean:
            contrib += cfg.weight_euclidean * max(0.0, euclid - cfg.offset_euclidean)
        if cfg.weight_chebyshev:
            contrib += cfg.weight_chebyshev * max(0.0, cheby - cfg.offset_chebyshev)
        if contrib > cfg.epsilon:
            total += contrib
    return 0.0 if total <= cfg.epsilon else float(total)


def heuristic_zero_threshold(config: Optional[HeuristicConfig] = None) -> float:
    cfg = config or get_heuristic_config()
    return cfg.epsilon


def heuristic_is_zero(value: float, config: Optional[HeuristicConfig] = None) -> bool:
    cfg = config or get_heuristic_config()
    return value <= cfg.epsilon


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
