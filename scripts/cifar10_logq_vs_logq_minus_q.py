from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class SmallCnn(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@dataclass
class EpochStats:
    epoch: int
    objective: str
    train_loss: float
    train_acc: float
    train_q_mean: float
    train_logq_mean: float
    train_logq_minus_q_mean: float
    test_loss: float
    test_acc: float
    test_q_mean: float
    test_logq_mean: float
    test_logq_minus_q_mean: float


def make_dataloaders(
    data_root: Path, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_ds = datasets.CIFAR10(
        root=str(data_root), train=True, transform=train_tf, download=True
    )
    test_ds = datasets.CIFAR10(
        root=str(data_root), train=False, transform=test_tf, download=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


def objective_loss(
    logits: torch.Tensor, targets: torch.Tensor, objective: str
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=-1)
    logq = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    q = logq.exp()

    if objective == "ce":
        # Minimize -log(q), i.e. maximize log(q).
        loss = -logq.mean()
    elif objective == "logq_minus_q":
        # Minimize -(log(q) - q) = -log(q) + q.
        loss = (-logq + q).mean()
    else:
        raise ValueError(f"Unknown objective: {objective}")

    return loss, q


def run_eval(
    model: nn.Module, loader: DataLoader, device: torch.device, objective: str
) -> tuple[float, float, float, float, float]:
    model.eval()
    total = 0
    total_correct = 0
    total_loss = 0.0
    total_q = 0.0
    total_logq = 0.0
    total_logq_minus_q = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss, q = objective_loss(logits, y, objective)
            preds = logits.argmax(dim=-1)

            bsz = y.size(0)
            total += bsz
            total_correct += (preds == y).sum().item()
            total_loss += loss.item() * bsz
            total_q += q.sum().item()
            logq = q.log()
            total_logq += logq.sum().item()
            total_logq_minus_q += (logq - q).sum().item()

    return (
        total_loss / max(total, 1),
        total_correct / max(total, 1),
        total_q / max(total, 1),
        total_logq / max(total, 1),
        total_logq_minus_q / max(total, 1),
    )


def train_one_objective(
    objective: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> list[EpochStats]:
    model = SmallCnn().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history: list[EpochStats] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        total_correct = 0
        total_loss = 0.0
        total_q = 0.0
        total_logq = 0.0
        total_logq_minus_q = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss, q = objective_loss(logits, y, objective)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            bsz = y.size(0)
            total += bsz
            total_correct += (preds == y).sum().item()
            total_loss += loss.item() * bsz
            total_q += q.sum().item()
            logq = q.log()
            total_logq += logq.sum().item()
            total_logq_minus_q += (logq - q).sum().item()

        scheduler.step()

        train_loss = total_loss / max(total, 1)
        train_acc = total_correct / max(total, 1)
        train_q_mean = total_q / max(total, 1)
        train_logq_mean = total_logq / max(total, 1)
        train_logq_minus_q_mean = total_logq_minus_q / max(total, 1)

        test_loss, test_acc, test_q_mean, test_logq_mean, test_logq_minus_q_mean = (
            run_eval(
                model=model,
                loader=test_loader,
                device=device,
                objective=objective,
            )
        )

        stats = EpochStats(
            epoch=epoch,
            objective=objective,
            train_loss=train_loss,
            train_acc=train_acc,
            train_q_mean=train_q_mean,
            train_logq_mean=train_logq_mean,
            train_logq_minus_q_mean=train_logq_minus_q_mean,
            test_loss=test_loss,
            test_acc=test_acc,
            test_q_mean=test_q_mean,
            test_logq_mean=test_logq_mean,
            test_logq_minus_q_mean=test_logq_minus_q_mean,
        )
        history.append(stats)
        print(
            f"[{objective}] epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f} test_q={test_q_mean:.4f}"
        )

    return history


def write_csv(history: list[EpochStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(EpochStats.__dataclass_fields__.keys()),
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row.__dict__)


def plot_results(history: list[EpochStats], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    objectives = sorted({h.objective for h in history})
    colors = {"ce": "#1f77b4", "logq_minus_q": "#d62728"}
    epochs = sorted({h.epoch for h in history})

    def series(metric: str, objective: str) -> list[float]:
        rows = [h for h in history if h.objective == objective]
        rows = sorted(rows, key=lambda x: x.epoch)
        return [getattr(r, metric) for r in rows]

    # Plot 1: accuracy.
    plt.figure(figsize=(8, 5))
    for obj in objectives:
        plt.plot(
            epochs,
            series("train_acc", obj),
            "--",
            color=colors.get(obj),
            label=f"{obj} train",
        )
        plt.plot(
            epochs,
            series("test_acc", obj),
            "-",
            color=colors.get(obj),
            label=f"{obj} test",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 Accuracy")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy.png", dpi=180)
    plt.close()

    # Plot 2: optimized train/test loss by objective.
    plt.figure(figsize=(8, 5))
    for obj in objectives:
        plt.plot(
            epochs,
            series("train_loss", obj),
            "--",
            color=colors.get(obj),
            label=f"{obj} train loss",
        )
        plt.plot(
            epochs,
            series("test_loss", obj),
            "-",
            color=colors.get(obj),
            label=f"{obj} test loss",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss (objective-specific)")
    plt.title("Objective Loss Curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "objective_loss.png", dpi=180)
    plt.close()

    # Plot 3: q statistics.
    plt.figure(figsize=(8, 5))
    for obj in objectives:
        plt.plot(
            epochs,
            series("train_q_mean", obj),
            "--",
            color=colors.get(obj),
            label=f"{obj} train q",
        )
        plt.plot(
            epochs,
            series("test_q_mean", obj),
            "-",
            color=colors.get(obj),
            label=f"{obj} test q",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Mean q(true_class)")
    plt.title("Mean True-Class Probability")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "q_mean.png", dpi=180)
    plt.close()

    # Plot 4: common objectives tracked on test set.
    plt.figure(figsize=(8, 5))
    for obj in objectives:
        plt.plot(
            epochs,
            series("test_logq_mean", obj),
            "-",
            color=colors.get(obj),
            label=f"{obj} test log(q)",
        )
        plt.plot(
            epochs,
            series("test_logq_minus_q_mean", obj),
            "--",
            color=colors.get(obj),
            label=f"{obj} test log(q)-q",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Mean value")
    plt.title("Test Objective Values")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "test_objectives.png", dpi=180)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 CNN with CE vs log(q)-q objective and save plots."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Where CIFAR-10 is downloaded/read.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./outputs/cifar10_logq_compare"),
        help="Where CSV and plots are written.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = make_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    all_history: list[EpochStats] = []
    for objective in ["ce", "logq_minus_q"]:
        history = train_one_objective(
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        all_history.extend(history)

    csv_path = args.out_dir / "metrics.csv"
    write_csv(all_history, csv_path)
    plot_results(all_history, args.out_dir)
    print(f"Saved metrics: {csv_path}")
    print(f"Saved plots under: {args.out_dir}")


if __name__ == "__main__":
    main()
