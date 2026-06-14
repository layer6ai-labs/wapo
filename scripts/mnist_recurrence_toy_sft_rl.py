from __future__ import annotations

import argparse
import copy
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TinyCausalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        ff: int = 256,
        max_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        h = self.tok(x) * (self.d_model**0.5) + self.pos(pos)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.enc(h, mask=mask)
        return self.out(h)


@dataclass
class TrainConfig:
    p: int
    n: int
    N: int
    a: int
    b: int
    img_side: int
    train_subset: int
    val_subset: int
    test_subset: int
    batch_size: int
    max_sft_steps: int
    sft_target_acc: float
    sft_eval_every: int
    sft_lr: float
    rl_steps: int
    rl_eval_every: int
    rl_lr: float
    num_workers: int
    seed: int
    data_root: Path
    out_dir: Path


@dataclass
class SftRecord:
    step: int
    sft_loss: float
    val_cls_acc: float
    val_seq_exact: float


@dataclass
class RlRecord:
    method: str
    step: int
    loss: float
    train_reward: float
    val_reward: float
    val_cls_acc: float
    val_seq_exact: float
    val_entropy: float


def make_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.Resize((cfg.img_side, cfg.img_side)), transforms.ToTensor()]
    )
    train_full = datasets.MNIST(
        root=str(cfg.data_root), train=True, download=True, transform=transform
    )
    test_full = datasets.MNIST(
        root=str(cfg.data_root), train=False, download=True, transform=transform
    )

    rng = np.random.RandomState(cfg.seed)
    train_idx = rng.choice(
        len(train_full), cfg.train_subset + cfg.val_subset, replace=False
    )
    train_split = train_idx[: cfg.train_subset]
    val_split = train_idx[cfg.train_subset :]
    test_idx = rng.choice(len(test_full), cfg.test_subset, replace=False)

    train_ds = Subset(train_full, train_split)
    val_ds = Subset(train_full, val_split)
    test_ds = Subset(test_full, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def image_to_tokens(x: torch.Tensor) -> torch.Tensor:
    return (x[:, 0] > 0.5).long().flatten(1)


def make_targets(y: torch.Tensor, p: int, n: int, a: int, b: int) -> torch.Tensor:
    out_len = n + 1
    t = torch.zeros(y.size(0), out_len, dtype=torch.long, device=y.device)
    t[:, 0] = y
    for i in range(1, out_len):
        t[:, i] = (a * t[:, i - 1] + b) % p
    return t


@torch.no_grad()
def class_acc_from_context(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        img_tok = image_to_tokens(x)
        logits = model(img_tok)[:, -1, :]
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def seq_exact_acc_greedy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    p: int,
    n: int,
    a: int,
    b: int,
) -> float:
    model.eval()
    exact = 0
    total = 0
    out_len = n + 1
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        img_tok = image_to_tokens(x)
        tgt = make_targets(y, p=p, n=n, a=a, b=b)
        cur = img_tok
        outs = []
        for _ in range(out_len):
            logits = model(cur)[:, -1, :]
            nxt = logits.argmax(dim=-1)
            outs.append(nxt)
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
        gen = torch.stack(outs, dim=1)
        exact += (gen == tgt).all(dim=1).sum().item()
        total += y.numel()
    return exact / max(total, 1)


def sft_train(
    cfg: TrainConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[nn.Module, list[SftRecord]]:
    img_len = cfg.img_side * cfg.img_side
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.sft_lr, weight_decay=1e-2)
    train_iter = iter(train_loader)
    step = 0
    history: list[SftRecord] = []

    while step < cfg.max_sft_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        img_tok = image_to_tokens(x)
        tgt = make_targets(y, p=cfg.p, n=cfg.n, a=cfg.a, b=cfg.b)
        seq = torch.cat([img_tok, tgt], dim=1)

        logits = model(seq[:, :-1])
        targets = seq[:, 1:]
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[:, img_len - 1 :] = True

        ce = F.cross_entropy(
            logits.reshape(-1, cfg.p),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        loss = ce[mask].mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % cfg.sft_eval_every == 0 or step == 1:
            val_cls = class_acc_from_context(model, val_loader, device)
            val_exact = seq_exact_acc_greedy(
                model, val_loader, device, p=cfg.p, n=cfg.n, a=cfg.a, b=cfg.b
            )
            row = SftRecord(
                step=step,
                sft_loss=float(loss.item()),
                val_cls_acc=val_cls,
                val_seq_exact=val_exact,
            )
            history.append(row)
            print(
                f"SFT step={step:4d} loss={row.sft_loss:.4f} "
                f"val_cls={val_cls:.4f} val_exact={val_exact:.4f}"
            )
            if val_cls >= cfg.sft_target_acc:
                print(
                    f"SFT early stop at step {step} "
                    f"(val_cls >= {cfg.sft_target_acc:.2f})"
                )
                break

    return model, history


@torch.no_grad()
def rollout_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    max_batches: int = 10,
) -> tuple[float, float]:
    out_len = cfg.n + 1
    model.eval()
    rewards = []
    entropies = []
    seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        img_tok = image_to_tokens(x)
        tgt = make_targets(y, p=cfg.p, n=cfg.n, a=cfg.a, b=cfg.b)
        cur = img_tok
        gen = []
        for _ in range(out_len):
            logits = model(cur)[:, -1, :]
            probs = logits.softmax(dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            tok = dist.sample()
            gen.append(tok)
            entropies.append(dist.entropy().mean().item())
            cur = torch.cat([cur, tok.unsqueeze(1)], dim=1)
        gen_t = torch.stack(gen, dim=1)
        # Binary reward: 1 only when the full sequence exactly matches.
        r = (gen_t == tgt).all(dim=1).float()
        rewards.append(r.mean().item())
        seen += 1
        if seen >= max_batches:
            break
    return float(np.mean(rewards)), float(np.mean(entropies))


def rl_step_loss(
    model: nn.Module,
    img_tok: torch.Tensor,
    tgt: torch.Tensor,
    method: str,
    N: int,
) -> tuple[torch.Tensor, float]:
    out_len = tgt.size(1)
    rewards = []
    logp_means = []

    for _ in range(N):
        cur = img_tok
        gen = []
        logps = []
        for _ in range(out_len):
            logits = model(cur)[:, -1, :]
            dist = torch.distributions.Categorical(logits=logits)
            tok = dist.sample()
            lp = dist.log_prob(tok)
            gen.append(tok)
            logps.append(lp)
            cur = torch.cat([cur, tok.unsqueeze(1)], dim=1)
        gen_t = torch.stack(gen, dim=1)
        lp_t = torch.stack(logps, dim=1).mean(dim=1)
        # Binary reward: exact sequence match only.
        r = (gen_t == tgt).all(dim=1).float()
        rewards.append(r)
        logp_means.append(lp_t)

    rewards_t = torch.stack(rewards, dim=0)
    logp_t = torch.stack(logp_means, dim=0)

    r_mean = rewards_t.mean(dim=0, keepdim=True)
    r_std = rewards_t.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)

    if method == "winner_only":
        raw = torch.where(rewards_t > r_mean, 1.0 - r_mean, torch.zeros_like(rewards_t))
        denom = (raw > 0).float().sum(dim=0, keepdim=True).clamp_min(1.0)
        w = raw / denom
    elif method == "maxrl":
        raw = torch.clamp(rewards_t - r_mean, min=0.0)
        denom = (raw > 0).float().sum(dim=0, keepdim=True).clamp_min(1.0)
        w = raw / denom
    elif method == "raft":
        raw = torch.clamp(rewards_t, min=0.0)
        denom = (raw > 0).float().sum(dim=0, keepdim=True).clamp_min(1.0)
        w = raw / denom
    elif method == "grpo":
        w = (rewards_t - r_mean) / r_std / float(N)
    else:
        raise ValueError(f"Unknown method: {method}")

    loss = -(w * logp_t).sum(dim=0).mean()
    return loss, rewards_t.mean().detach().item()


def run_rl(
    cfg: TrainConfig,
    method: str,
    sft_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[nn.Module, list[RlRecord]]:
    model = copy.deepcopy(sft_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.rl_lr, weight_decay=1e-2)
    train_iter = iter(train_loader)
    history: list[RlRecord] = []

    for step in range(1, cfg.rl_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        img_tok = image_to_tokens(x)
        tgt = make_targets(y, p=cfg.p, n=cfg.n, a=cfg.a, b=cfg.b)

        loss, train_reward = rl_step_loss(model, img_tok, tgt, method=method, N=cfg.N)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg.rl_eval_every == 0 or step == 1:
            cls = class_acc_from_context(model, val_loader, device)
            seq_exact = seq_exact_acc_greedy(
                model, val_loader, device, p=cfg.p, n=cfg.n, a=cfg.a, b=cfg.b
            )
            val_reward, ent = rollout_stats(
                model, val_loader, device, cfg, max_batches=8
            )
            rec = RlRecord(
                method=method,
                step=step,
                loss=float(loss.item()),
                train_reward=float(train_reward),
                val_reward=val_reward,
                val_cls_acc=cls,
                val_seq_exact=seq_exact,
                val_entropy=ent,
            )
            history.append(rec)
            print(
                f"[{method}] step={step:4d} loss={rec.loss:.4f} "
                f"train_r={rec.train_reward:.4f} val_r={val_reward:.4f} val_cls={cls:.4f}"
            )

    return model, history


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_all(
    out_dir: Path,
    sft_hist: list[SftRecord],
    rl_hist: list[RlRecord],
    methods: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if sft_hist:
        sft_steps = [x.step for x in sft_hist]
        sft_cls = [x.val_cls_acc for x in sft_hist]
        sft_exact = [x.val_seq_exact for x in sft_hist]
        plt.figure(figsize=(8, 4))
        plt.plot(sft_steps, sft_cls, marker="o", label="SFT val cls acc")
        plt.plot(sft_steps, sft_exact, marker="o", label="SFT val seq exact")
        plt.xlabel("SFT step")
        plt.ylabel("Metric")
        plt.title("SFT Progress")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "sft_progress.png", dpi=180)
        plt.close()

    if rl_hist:
        df = {
            "method": [x.method for x in rl_hist],
            "step": [x.step for x in rl_hist],
            "val_reward": [x.val_reward for x in rl_hist],
            "val_cls_acc": [x.val_cls_acc for x in rl_hist],
            "val_seq_exact": [x.val_seq_exact for x in rl_hist],
            "val_entropy": [x.val_entropy for x in rl_hist],
        }
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
        plots = [
            ("val_reward", "Validation rollout reward"),
            ("val_cls_acc", "Validation class accuracy"),
            ("val_seq_exact", "Validation sequence exact-match"),
            ("val_entropy", "Validation policy entropy"),
        ]
        for (metric, title), ax in zip(plots, axes.flatten(), strict=True):
            for method in methods:
                idx = [i for i, m in enumerate(df["method"]) if m == method]
                xs = [df["step"][i] for i in idx]
                ys = [df[metric][i] for i in idx]
                ax.plot(xs, ys, marker="o", label=method)
            ax.set_title(title)
            ax.set_xlabel("RL step")
            ax.grid(alpha=0.25)
        axes[0, 0].legend()
        plt.tight_layout()
        plt.savefig(out_dir / "rl_metrics.png", dpi=180)
        plt.close()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description=(
            "MNIST toy experiment: SFT on recurrence task then RL with "
            "winner_only/maxrl/raft/grpo."
        )
    )
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--a", type=int, default=3)
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--img-side", type=int, default=14)
    parser.add_argument("--train-subset", type=int, default=12000)
    parser.add_argument("--val-subset", type=int, default=2000)
    parser.add_argument("--test-subset", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-sft-steps", type=int, default=1000)
    parser.add_argument("--sft-target-acc", type=float, default=0.60)
    parser.add_argument("--sft-eval-every", type=int, default=100)
    parser.add_argument("--sft-lr", type=float, default=2e-3)
    parser.add_argument("--rl-steps", type=int, default=400)
    parser.add_argument("--rl-eval-every", type=int, default=50)
    parser.add_argument("--rl-lr", type=float, default=8e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("./outputs/mnist_recurrence_toy_sft_rl")
    )
    args = parser.parse_args()
    return TrainConfig(
        p=args.p,
        n=args.n,
        N=args.N,
        a=args.a,
        b=args.b,
        img_side=args.img_side,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        test_subset=args.test_subset,
        batch_size=args.batch_size,
        max_sft_steps=args.max_sft_steps,
        sft_target_acc=args.sft_target_acc,
        sft_eval_every=args.sft_eval_every,
        sft_lr=args.sft_lr,
        rl_steps=args.rl_steps,
        rl_eval_every=args.rl_eval_every,
        rl_lr=args.rl_lr,
        num_workers=args.num_workers,
        seed=args.seed,
        data_root=args.data_root,
        out_dir=args.out_dir,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = ["winner_only", "maxrl", "raft", "grpo"]

    print(f"Using device: {device}")
    print(
        {
            "p": cfg.p,
            "n": cfg.n,
            "N": cfg.N,
            "max_sft_steps": cfg.max_sft_steps,
            "rl_steps": cfg.rl_steps,
        }
    )

    train_loader, val_loader, test_loader = make_loaders(cfg)
    max_len = cfg.img_side * cfg.img_side + (cfg.n + 1)
    model = TinyCausalTransformer(vocab_size=cfg.p, max_len=max_len).to(device)

    sft_model, sft_hist = sft_train(cfg, model, train_loader, val_loader, device)
    sft_test_cls = class_acc_from_context(sft_model, test_loader, device)
    sft_test_exact = seq_exact_acc_greedy(
        sft_model, test_loader, device, p=cfg.p, n=cfg.n, a=cfg.a, b=cfg.b
    )
    print(f"SFT test class acc: {sft_test_cls:.4f}")
    print(f"SFT test sequence exact: {sft_test_exact:.4f}")

    all_rl: list[RlRecord] = []
    summary_rows = []
    for method in methods:
        model_m, hist_m = run_rl(
            cfg, method, sft_model, train_loader, val_loader, device
        )
        all_rl.extend(hist_m)
        test_cls = class_acc_from_context(model_m, test_loader, device)
        test_exact = seq_exact_acc_greedy(
            model_m, test_loader, device, p=cfg.p, n=cfg.n, a=cfg.a, b=cfg.b
        )
        test_rollout_reward = rollout_stats(model_m, test_loader, device, cfg, 12)[0]
        summary_rows.append(
            {
                "method": method,
                "test_cls_acc": test_cls,
                "test_seq_exact": test_exact,
                "test_rollout_reward": test_rollout_reward,
            }
        )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(cfg.out_dir / "sft_history.csv", [x.__dict__ for x in sft_hist])
    write_csv(cfg.out_dir / "rl_history.csv", [x.__dict__ for x in all_rl])
    write_csv(cfg.out_dir / "summary.csv", summary_rows)
    plot_all(cfg.out_dir, sft_hist, all_rl, methods)

    print("Final summary:")
    for row in sorted(
        summary_rows, key=lambda x: x["test_rollout_reward"], reverse=True
    ):
        print(row)
    print(f"Saved outputs to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
