from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.greedy_dataset import GreedyTeacherDataset
from models.policy import ModelConfig, PolicyCNN, PolicyRetNet


@dataclass
class TrainConfig:
    dataset: Path
    size: int
    architecture: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    dropout: float
    embed_dim: int
    hidden_channels: int
    depth: int
    d_model: int
    num_layers: int
    num_heads: int
    device: str
    seed: int
    augment_rotations: bool
    use_wandb: bool
    wandb_project: str | None
    wandb_entity: str | None
    save_dir: Path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    base_dataset = GreedyTeacherDataset(cfg.dataset, size=cfg.size, augment_rotations=False)
    meta = base_dataset.describe()

    indices = list(range(len(base_dataset)))
    random.Random(cfg.seed).shuffle(indices)
    split = max(1, int(0.9 * len(indices)))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = GreedyTeacherDataset(
        cfg.dataset,
        size=cfg.size,
        augment_rotations=cfg.augment_rotations,
        seed=cfg.seed,
        indices=train_indices,
    )
    val_dataset = GreedyTeacherDataset(
        cfg.dataset,
        size=cfg.size,
        augment_rotations=False,
        seed=cfg.seed + 1,
        indices=val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader, meta


def build_model(cfg: TrainConfig, meta: Dict[str, int]) -> nn.Module:
    model_cfg = ModelConfig(
        size=cfg.size,
        vocab_size=meta["vocab_size"],
        num_ops=meta["num_ops"],
        embed_dim=cfg.embed_dim,
    )
    arch = cfg.architecture.lower()
    if arch == "cnn":
        model = PolicyCNN(
            model_cfg,
            hidden_channels=cfg.hidden_channels,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )
    elif arch == "retnet":
        model = PolicyRetNet(
            model_cfg,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {cfg.architecture}")
    return model


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_exact = 0
    total_samples = 0
    with torch.no_grad():
        for grids, labels, targets in loader:
            grids = grids.to(device)
            labels = labels.to(device)
            targets = targets.to(device)
            logits = model(grids)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += float(loss.item() * grids.size(0))
            hits = (targets[torch.arange(grids.size(0), device=device), preds] > 0).float().sum().item()
            total_correct += float(hits)
            total_exact += int((preds == labels).sum().item())
            total_samples += grids.size(0)
    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    exact = total_exact / max(1, total_samples)
    return avg_loss, acc, exact


def train(cfg: TrainConfig) -> Dict[str, float]:
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    train_loader, val_loader, meta = build_dataloaders(cfg)
    model = build_model(cfg, meta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    wandb_run = None
    if cfg.use_wandb:
        import wandb

        wandb_kwargs = {
            "project": cfg.wandb_project or os.environ.get("WANDB_PROJECT", "procon-policy"),
            "config": {
                **meta,
                **asdict(cfg),
            },
        }
        if cfg.wandb_entity:
            wandb_kwargs["entity"] = cfg.wandb_entity
        wandb_run = wandb.init(**wandb_kwargs)

    best_val_acc = 0.0
    best_val_exact = 0.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        running_samples = 0
        for grids, labels, targets in train_loader:
            grids = grids.to(device)
            labels = labels.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(grids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += float(loss.item() * grids.size(0))
            hits = (targets[torch.arange(grids.size(0), device=device), preds] > 0).float().sum().item()
            running_correct += float(hits)
            running_samples += grids.size(0)
        scheduler.step()

        train_loss = running_loss / max(1, running_samples)
        train_acc = running_correct / max(1, running_samples)
        val_loss, val_acc, val_exact = evaluate(model, val_loader, device, criterion)

        if val_exact >= best_val_exact:
            best_val_acc = val_acc
            best_val_exact = val_exact
            best_state = {
                "model_state": model.state_dict(),
                "meta": meta,
                "config": asdict(cfg),
            }

        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_exact": val_exact,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        print(
            f"epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_exact={val_exact:.3f}"
        )

    if wandb_run:
        wandb_run.summary.update({
            "best_val_acc": best_val_acc,
            "best_val_exact": best_val_exact,
        })
        wandb_run.finish()

    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        arch = cfg.architecture.lower()
        save_path = cfg.save_dir / f"{arch}_size{cfg.size}.pth"
        torch.save(best_state, save_path)
        print(f"saved best model to {save_path}")

    return {"best_val_acc": best_val_acc}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train policy network to mimic greedy teacher")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/datasets/manhattan_greedy_4x4.jsonl"))
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--architecture", type=str, choices=["cnn", "retnet"], default="cnn")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment_rotations", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--save_dir", type=Path, default=Path("artifacts/models"))

    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
