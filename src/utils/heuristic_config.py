from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class HeuristicConfig:
    weight_euclidean: float = 1.0
    weight_chebyshev: float = 0.0
    offset_euclidean: float = 1.0
    offset_chebyshev: float = 1.0
    epsilon: float = 1e-9


def _find_config_path(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit is not None:
        return explicit if explicit.exists() else None
    for env_key in ("HEURISTIC_CONFIG_PATH", "WASTAR_HEURISTIC_CONFIG"):
        env_val = os.getenv(env_key)
        if env_val:
            candidate = Path(env_val)
            if candidate.exists():
                return candidate
    default = Path("heuristic.yaml")
    return default if default.exists() else None


def _parse_simple_yaml(text: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    section: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":") and ":" not in line[:-1]:
            section = line[:-1].strip()
            continue
        if ":" not in line:
            continue
        key_part, value_part = line.split(":", 1)
        key = key_part.strip()
        value_str = value_part.strip()
        if value_str and value_str[0] in {'"', '\''} and value_str[-1] == value_str[0]:
            value_str = value_str[1:-1]
        try:
            value = float(value_str)
        except ValueError:
            continue
        full_key = f"{section}.{key}" if section else key
        values[full_key] = value
    return values


def load_heuristic_config(path: Optional[Path] = None) -> HeuristicConfig:
    base = HeuristicConfig()
    config_path = _find_config_path(path)
    if config_path is None:
        return base
    try:
        parsed = _parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    except OSError:
        return base

    data = {
        key.replace("_", "."): value for key, value in parsed.items()
    }

    weight_euclidean = data.get("weights.euclidean", base.weight_euclidean)
    weight_chebyshev = data.get("weights.chebyshev", base.weight_chebyshev)
    offset_euclidean = data.get("offsets.euclidean", base.offset_euclidean)
    offset_chebyshev = data.get("offsets.chebyshev", base.offset_chebyshev)
    epsilon = max(1e-12, data.get("epsilon", base.epsilon))

    return HeuristicConfig(
        weight_euclidean=weight_euclidean,
        weight_chebyshev=weight_chebyshev,
        offset_euclidean=offset_euclidean,
        offset_chebyshev=offset_chebyshev,
        epsilon=epsilon,
    )


@lru_cache(maxsize=1)
def get_heuristic_config() -> HeuristicConfig:
    return load_heuristic_config()


def reload_heuristic_config(path: Optional[Path] = None) -> HeuristicConfig:
    get_heuristic_config.cache_clear()  # type: ignore[attr-defined]
    return load_heuristic_config(path)
