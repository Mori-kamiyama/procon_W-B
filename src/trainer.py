from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from utils.io import list_problem_files, save_solution
from utils.metrics import pairs_rate
from run_search import ensure_cli_solver, run_one_problem


def _maybe_import_wandb():
    try:
        import wandb  # type: ignore

        return wandb
    except Exception:
        class _Dummy:
            def init(self, *a, **k):
                class _Run:
                    config = {}

                    class summary(dict):
                        def update(self, *a, **k):
                            dict.update(self, *a, **k)

                    def log(self, *a, **k):
                        pass

                return _Run()

            def log(self, *a, **k):
                pass

        return _Dummy()


@dataclass
class Config:
    project: str = "procon-wastar"
    size: int = 6
    w: float = 2.0
    max_problems: int = 5
    time_limit_s: Optional[float] = None
    problems_root: str = "problems"
    solver_bin: str = "core/wastar"
    save_dir: str = "artifacts/results"
    fast_max_small_n: Optional[int] = None
    fast_cand_cap: Optional[int] = None
    fast_topk: Optional[int] = None
    alpha: Optional[float] = None
    tie_break: Optional[str] = None  # h_min/h_max/g_min/g_max
    max_depth: Optional[int] = None
    k_top_moves: Optional[int] = None


def train(cfg: Config | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(cfg, dict):
        cfg = Config(**cfg)

    wandb = _maybe_import_wandb()
    import os
    project_name = os.environ.get("WANDB_PROJECT", cfg.project)
    entity = os.environ.get("WANDB_ENTITY")
    init_kwargs = {"project": project_name}
    if entity:
        init_kwargs["entity"] = entity
    run = wandb.init(**init_kwargs, config={
        "size": cfg.size,
        "w": cfg.w,
        "max_problems": cfg.max_problems,
        "time_limit_s": cfg.time_limit_s,
        "fast_max_small_n": cfg.fast_max_small_n,
        "fast_cand_cap": cfg.fast_cand_cap,
        "fast_topk": cfg.fast_topk,
        "alpha": cfg.alpha,
        "tie_break": cfg.tie_break,
        "max_depth": cfg.max_depth,
        "k_top_moves": cfg.k_top_moves,
    })

    # Merge sweep config (if any) from W&B
    try:
        wbconf = dict(run.config)
        for k, v in wbconf.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
    except Exception:
        pass

    # Friendly naming on W&B
    try:
        name_parts = [f"size{cfg.size}", f"w{cfg.w}"]
        if cfg.alpha is not None:
            name_parts.append(f"a{cfg.alpha}")
        if cfg.tie_break is not None:
            name_parts.append(f"tb-{cfg.tie_break}")
        run.name = "-".join(map(str, name_parts))
        run.group = f"size-{cfg.size}"
        run.tags = ["wastar", f"size{cfg.size}"]
    except Exception:
        pass

    solver_bin = Path(cfg.solver_bin)
    ensure_cli_solver(solver_bin)

    files = list_problem_files(cfg.problems_root, cfg.size, cfg.max_problems)
    fast_params = {}
    if cfg.fast_max_small_n is not None:
        fast_params["FAST_MAX_SMALL_N"] = cfg.fast_max_small_n
    if cfg.fast_cand_cap is not None:
        fast_params["FAST_CAND_CAP"] = cfg.fast_cand_cap
    if cfg.fast_topk is not None:
        fast_params["FAST_TOPK"] = cfg.fast_topk
    fast_params = fast_params or None

    save_root = Path(cfg.save_dir) / f"{cfg.size}x{cfg.size}"
    save_root.mkdir(parents=True, exist_ok=True)

    import statistics as stats

    time_list = []
    pr_list = []
    ops_list = []
    open_max_list = []
    nodes_expanded_list = []
    nodes_generated_list = []
    ok_count = 0

    table_rows = []

    for p in files:
        res = run_one_problem(
            solver=solver_bin,
            problem_path=p,
            w_weight=cfg.w,
            time_limit_s=cfg.time_limit_s,
            fast_params=fast_params,
            alpha=cfg.alpha,
            tie_break=cfg.tie_break,
            max_depth=cfg.max_depth,
            k_top_moves=cfg.k_top_moves,
        )
        time_list.append(res["time_s"])
        pr_list.append(res["pairs_rate"])
        ops_list.append(res["ops_count"])
        ok_count += 1 if res["ok"] else 0

        # per-problem log
        run.log({
            "problem": str(p),
            "time_s": res["time_s"],
            "pairs_rate": res["pairs_rate"],
            "ops_count": res["ops_count"],
            "timeout": res["timeout"],
            "solved": res.get("solved"),
            "partial": res.get("partial"),
            "open_max": res.get("open_max"),
            "nodes_expanded": res.get("nodes_expanded"),
            "nodes_generated": res.get("nodes_generated"),
        })
        if res.get("open_max") is not None:
            open_max_list.append(int(res["open_max"]))
        if res.get("nodes_expanded") is not None:
            nodes_expanded_list.append(int(res["nodes_expanded"]))
        if res.get("nodes_generated") is not None:
            nodes_generated_list.append(int(res["nodes_generated"]))
        # Save artifact-like JSON for reproduction
        out_path = save_root / (p.stem + ".solution.json")
        save_solution(out_path, res["ops"])  # minimal result
        table_rows.append({
            "problem": str(p),
            "time_s": res["time_s"],
            "pairs_rate": res["pairs_rate"],
            "ops_count": res["ops_count"],
            "ok": res["ok"],
        })

    summary = {
        "problems_evaluated": len(files),
        "ok_count": ok_count,
        "pairs_rate_avg": float(stats.mean(pr_list)) if pr_list else 0.0,
        "ops_avg": float(stats.mean([x for x in ops_list if x is not None])) if ops_list else 0.0,
        "time_avg_s": float(stats.mean(time_list)) if time_list else 0.0,
        "open_max_max": int(max(open_max_list)) if open_max_list else 0,
        "nodes_expanded_avg": float(stats.mean(nodes_expanded_list)) if nodes_expanded_list else None,
        "nodes_generated_avg": float(stats.mean(nodes_generated_list)) if nodes_generated_list else None,
    }
    # update W&B summary if available
    try:
        # Log problems table
        cols = ["problem", "time_s", "pairs_rate", "ops_count", "ok"]
        try:
            import wandb as _wb

            table = _wb.Table(columns=cols)
            for r in table_rows:
                table.add_data(*[r.get(c) for c in cols])
            run.log({"problems_table": table})
        except Exception:
            pass

        # Attach artifacts: problems list and solutions dir
        problems_txt = save_root / "problems_used.txt"
        problems_txt.write_text("\n".join([str(p) for p in files]), encoding="utf-8")

        try:
            import wandb as _wb

            a = _wb.Artifact(
                name=f"solutions-{cfg.size}x{cfg.size}-{run.id}", type="solutions",
                metadata={"size": cfg.size, "w": cfg.w, "alpha": cfg.alpha,
                          "tie_break": cfg.tie_break, "time_limit_s": cfg.time_limit_s}
            )
            a.add_dir(str(save_root))
            a.add_file(str(problems_txt), name="problems_used.txt")
            run.log_artifact(a)
        except Exception:
            pass

        run.summary.update(summary)
    except Exception:
        pass
    return summary


if __name__ == "__main__":
    # Minimal local run
    cfg = Config()
    s = train(cfg)
    print("SUMMARY:", json.dumps(s, ensure_ascii=False))
