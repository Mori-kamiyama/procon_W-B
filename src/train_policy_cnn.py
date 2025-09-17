"""Train a convolutional policy network on teacher data."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


BoardTensor = torch.Tensor


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


class MoveSpace:
    """Utility for mapping (x, y, n) moves to categorical labels."""

    def __init__(self, size: int, min_n: int = 2, max_n: int | None = None):
        self.size = size
        self.min_n = min_n
        self.max_n = max_n or size

        self.index_to_move: List[Tuple[int, int, int]] = []
        for n in range(self.min_n, self.max_n + 1):
            span = size - n + 1
            if span <= 0:
                continue
            for y in range(span):
                for x in range(span):
                    self.index_to_move.append((x, y, n))

        self.move_to_index = {move: idx for idx, move in enumerate(self.index_to_move)}

    def encode(self, x: int, y: int, n: int) -> int:
        try:
            return self.move_to_index[(x, y, n)]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Move ({x}, {y}, {n}) is outside the move space") from exc

    def decode(self, index: int) -> Tuple[int, int, int]:
        return self.index_to_move[index]

    @property
    def num_moves(self) -> int:
        return len(self.index_to_move)


class TeacherDataset(Dataset[Tuple[BoardTensor, int]]):
    def __init__(
        self,
        *,
        path: Path,
        size: int,
        value_count: int,
        move_space: MoveSpace,
    ) -> None:
        self.path = path
        self.size = size
        self.value_count = value_count
        self.move_space = move_space

        self._boards: List[BoardTensor] = []
        self._labels: List[int] = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                grid = torch.tensor(obj["grid"], dtype=torch.long)
                if grid.shape != (size, size):
                    raise ValueError(
                        f"Expected {size}x{size} grid, got {tuple(grid.shape)} from sample"
                    )
                board = torch.nn.functional.one_hot(grid, num_classes=value_count)
                board = board.permute(2, 0, 1).to(torch.float32)
                move = obj["move"]
                label = move_space.encode(int(move["x"]), int(move["y"]), int(move["n"]))
                self._boards.append(board)
                self._labels.append(label)

        if not self._boards:
            raise RuntimeError(f"No samples found in {path}")

    def __len__(self) -> int:
        return len(self._boards)

    def __getitem__(self, index: int) -> Tuple[BoardTensor, int]:
        return self._boards[index], self._labels[index]


class PolicyCNN(nn.Module):
    def __init__(self, in_channels: int, board_size: int, num_classes: int):
        super().__init__()
        hidden = 128
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(hidden * board_size * board_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: BoardTensor) -> BoardTensor:
        return self.layers(x)


@dataclass
class TrainConfig:
    dataset: Path
    size: int = 6
    value_count: int = 18
    min_n: int = 2
    max_n: int | None = None
    batch_size: int = 128
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_split: float = 0.9
    seed: int = 42
    num_workers: int = 0
    device: torch.device = _default_device()
    save_path: Path | None = None


def _split_dataset(
    dataset: Dataset[Tuple[BoardTensor, int]],
    train_ratio: float,
    seed: int,
) -> Tuple[Dataset[Tuple[BoardTensor, int]], Dataset[Tuple[BoardTensor, int]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    total = len(dataset)
    train_size = max(1, int(total * train_ratio))
    if train_size >= total:
        train_size = total - 1
    val_size = total - train_size
    if val_size <= 0:
        raise ValueError("Validation split produced no samples; adjust train_split")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.numel()


def train(config: TrainConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    move_space = MoveSpace(config.size, config.min_n, config.max_n)
    dataset = TeacherDataset(
        path=config.dataset,
        size=config.size,
        value_count=config.value_count,
        move_space=move_space,
    )

    train_ds, val_ds = _split_dataset(dataset, config.train_split, config.seed)

    loader_kwargs = {"batch_size": config.batch_size, "num_workers": config.num_workers, "shuffle": True}
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, num_workers=config.num_workers)

    model = PolicyCNN(
        in_channels=config.value_count,
        board_size=config.size,
        num_classes=move_space.num_moves,
    ).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        batches = 0
        for boards, labels in train_loader:
            boards = boards.to(config.device)
            labels = labels.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(boards)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += _accuracy(logits.detach(), labels)
            batches += 1

        train_loss = total_loss / max(1, batches)
        train_acc = total_acc / max(1, batches)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for boards, labels in val_loader:
                boards = boards.to(config.device)
                labels = labels.to(config.device)
                logits = model(boards)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_acc += _accuracy(logits, labels)
                val_batches += 1

        val_loss = val_loss / max(1, val_batches)
        val_acc = val_acc / max(1, val_batches)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if config.save_path and val_acc >= best_val_acc:
            best_val_acc = val_acc
            config.save_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "move_space": move_space.index_to_move,
                "val_acc": val_acc,
            }
            torch.save(checkpoint, config.save_path)


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to teacher JSONL dataset")
    parser.add_argument("--size", type=int, default=6, help="Board size (default: 6)")
    parser.add_argument("--value-count", type=int, default=18, help="Number of distinct tile values")
    parser.add_argument("--min-n", type=int, default=2, help="Minimum rotation size to model")
    parser.add_argument("--max-n", type=int, default=None, help="Maximum rotation size to model (default: size)")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for the AdamW optimizer"
    )
    parser.add_argument(
        "--train-split", type=float, default=0.9, help="Fraction of samples to use for training"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional path to save the best model checkpoint (.pt)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (cpu, cuda, mps). Defaults to auto-detect."
    )

    args = parser.parse_args(argv)

    device = torch.device(args.device) if args.device else _default_device()

    return TrainConfig(
        dataset=args.dataset,
        size=args.size,
        value_count=args.value_count,
        min_n=args.min_n,
        max_n=args.max_n,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
        save_path=args.save_path,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    print(f"Using device: {config.device}")
    print(f"Training samples: {config.dataset}")
    train(config)


if __name__ == "__main__":
    main()
