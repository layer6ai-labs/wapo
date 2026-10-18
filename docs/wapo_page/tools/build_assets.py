"""Rebuild publication figures and the local W&B evidence bundle.

Run from the research workspace with a Python environment containing numpy,
matplotlib and wandb: python wapo/docs/wapo_page/tools/build_assets.py
No network requests or credentials are used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import struct
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SITE = Path(__file__).resolve().parents[1]
WORKSPACE = SITE.parents[2]
PAPER = WORKSPACE / "EMNLP_2026/wapo_paper/plots"
FIG = SITE / "assets/figures"
DATA = SITE / "assets/data"
INK = "#182b31"
TEAL = "#16766c"
ORANGE = "#cf673f"
BLUE = "#4776a7"
GOLD = "#b78a28"
COLORS = {
    "Pos-peak": TEAL,
    "Pos-valley": GOLD,
    "Neg-peak": ORANGE,
    "Neg-valley": "#a95f87",
    "Pos": TEAL,
    "Neg": ORANGE,
    "Peak": BLUE,
    "Valley": GOLD,
    "DAPO": "#9aa5a8",
    "GRPO": BLUE,
    "GSPO": GOLD,
    "WAPO": TEAL,
    "Base": "#a0a0a0",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": "#607076",
        "ytick.color": "#607076",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#ccd3d2",
        "axes.titleweight": "normal",
        "svg.fonttype": "none",
        "svg.hashsalt": "wapo-story",
        "savefig.facecolor": "white",
    }
)


def finish(fig, name):
    output = FIG / f"{name}.svg"
    fig.savefig(output, bbox_inches="tight", metadata={"Date": None})
    output.write_text(
        "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def style(ax, ylabel, title="", xmax=None):
    ax.set(ylabel=ylabel, xlabel="Gradient steps", title=title)
    if xmax:
        ax.set_xlim(0, xmax)
    ax.grid(axis="y", color="#e7e9e6", linewidth=0.7)
    ax.set_axisbelow(True)


def smooth(values, window):
    v = np.asarray(values, dtype=float)
    return np.array([v[max(0, i - window + 1) : i + 1].mean() for i in range(len(v))])


def csvrows(name):
    with (PAPER / "data" / name).open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def csvplot(ax, filename, label, metric, window=1, xmax=None, **kwargs):
    rows = csvrows(filename)
    if xmax is not None:
        rows = [row for row in rows if row[next(iter(row))] <= xmax]
    x = np.array([r[next(iter(r))] for r in rows])
    y = np.array(
        [
            r[metric] if isinstance(metric, str) else list(r.values())[metric]
            for r in rows
        ]
    )
    ax.plot(x, smooth(y, window), label=label, linewidth=1.9, **kwargs)
    if xmax:
        ax.set_xlim(0, xmax)


def entropy_control_plot():
    """Reflow the OTT-Qwen and Numina-Gemma panels from the entropy-control study."""
    source = (
        WORKSPACE
        / "neurips_2026/rlvr_entropy_control_overleaf/plots/entropy-filtering-reward-entropy"
    )
    experiments = [
        (
            "ott-qwen",
            "Qwen3-4B",
            "OTT-QA",
            0.1,
            400,
            0.65,
            12.5,
            ["s7veiho6", "q5fkevcu", "wugi6j1j"],
        ),
        (
            "numina-gemma",
            "Gemma3-4B",
            "NuminaMath-LEAN",
            0.2,
            600,
            0.62,
            13,
            ["ne4g8sig", "ou6gb57a", "ghv2o7su"],
        ),
    ]
    filters = [
        ("baseline", "Baseline", BLUE),
        ("upper", r"Mask $\Delta H > \tau^{+}$", TEAL),
        ("two-sided", r"Also mask $\Delta H < \tau^{-}$", ORANGE),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 6.2), layout="constrained")
    panels = []
    for row, (
        prefix,
        model,
        dataset,
        upper,
        horizon,
        reward_max,
        entropy_max,
        runs,
    ) in enumerate(experiments):
        series = {}
        for (suffix, label, color), run_id in zip(filters, runs):
            path = source / "data" / f"{prefix}-{suffix}.csv"
            with path.open() as file:
                values = [
                    {key: float(value) for key, value in item.items()}
                    for item in csv.DictReader(file)
                ]
            values = [item for item in values if item["gradient_step"] <= horizon]
            series[suffix] = {
                "run_id": run_id,
                "source_csv": str(path.relative_to(WORKSPACE)),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "values": values,
            }
            for col, metric in enumerate(["reward", "entropy"]):
                axes[row, col].plot(
                    [item["gradient_step"] for item in values],
                    [item[metric] for item in values],
                    label=label,
                    color=color,
                    linewidth=1.9,
                )
        style(
            axes[row, 0],
            "Training F1 reward" if row == 0 else "Training reward",
            f"{model} · {dataset}",
            xmax=horizon,
        )
        style(
            axes[row, 1],
            "Policy entropy (nats)",
            "Entropy during training",
            xmax=horizon,
        )
        axes[row, 0].set_ylim(0, reward_max)
        axes[row, 1].set_ylim(0, entropy_max)
        for ax in axes[row]:
            ax.set_xticks([0, 100, 200, 300, 400] if row == 0 else [0, 200, 400, 600])
        panels.append(
            {
                "model": model,
                "dataset": dataset,
                "objective": "DAPO",
                "tau_plus": upper,
                "tau_minus": -0.1,
                "displayed_steps": [0, horizon],
                "series": series,
            }
        )
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    finish(fig, "entropy-control")
    (DATA / "entropy-control.json").write_text(
        json.dumps(
            {
                "source_plot": str((source / "plot.tex").relative_to(WORKSPACE)),
                "source_pdf": "neurips_2026/131_Diagnosing_and_Stabilizing.pdf",
                "source_figure": 2,
                "panels": panels,
            },
            indent=2,
        )
        + "\n"
    )


def clipping_plots():
    """Keep each displayed comparison explicit about its model and loss."""
    for model, loss, prefix, suffixes, name in [
        ("Gemma3-4B", "DAPO", "ott_gemma_drdapo_10_", ["0_8", "0_1"], "clipping-gemma"),
        ("Qwen3-4B", "GRPO", "ott_qwen_grpo_eps_", ["0_2", "9_0"], "clipping-qwen"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.4), layout="constrained")
        labels = (
            ["ε low = 0.2", "ε low = 0.9"] if loss == "DAPO" else ["ε = 0.2", "ε = 9"]
        )
        for suffix, label, color in zip(suffixes, labels, [ORANGE, TEAL]):
            for ax, metric in zip(axes, ["f1_reward", "entropy"]):
                csvplot(
                    ax,
                    prefix + suffix + ".csv",
                    label,
                    metric,
                    window=1 if loss == "GRPO" else 10,
                    xmax=600 if loss == "GRPO" else None,
                    color=color,
                )
        style(axes[0], "Training F1 reward", f"{model} · OTT-QA · {loss}")
        style(axes[1], "Policy entropy (nats)", "Entropy during training")
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="outside lower center",
            ncol=2,
            frameon=False,
        )
        finish(fig, name)
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.3), layout="constrained")
    for suffix, label, col in [
        ("10-0.1", "ε low = 0.9 · looser", TEAL),
        ("10-0.8", "ε low = 0.2 · tighter", ORANGE),
    ]:
        for ax, m in zip(axes, ["f1_reward", "entropy"]):
            csvplot(
                ax, f"ott-smol-dr_dapo-{suffix}_{m}.csv", label, 1, window=10, color=col
            )
    style(axes[0], "Training F1 reward", "Keep more divergent updates")
    style(axes[1], "Policy entropy (nats)", "Entropy under the two tolerances")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    finish(fig, "clipping-smol")
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.4), layout="constrained")
    for ax, model, prefix, suffixes in [
        (axes[0], "Qwen3-4B · OTT-QA", "ott_qwen_grpo_eps_", ["0_2", "9_0"]),
        (axes[1], "Gemma3-4B · OTT-QA", "ott_gemma_drdapo_10_", ["0_8", "0_1"]),
    ]:
        for suffix, label, col in zip(
            suffixes, ["Tighter clipping", "Looser clipping"], [ORANGE, TEAL]
        ):
            rows = csvrows(prefix + suffix + ".csv")
            keys = list(rows[0])
            metric = next((k for k in keys if "f1" in k), keys[1])
            csvplot(
                ax,
                prefix + suffix + ".csv",
                label,
                metric,
                window=1 if "Qwen" in model else 10,
                xmax=600 if "Qwen" in model else None,
                color=col,
            )
        style(ax, "Training F1 reward", model)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    finish(fig, "clipping-additional")


def verify_clipping_sources():
    """Match all displayed CSV rows to their local W&B event histories."""
    sources = [
        ("2uir7dzw", "ott_gemma_drdapo_10_0_8.csv", "dr_dapo", None),
        ("8deer41o", "ott_gemma_drdapo_10_0_1.csv", "dr_dapo", None),
        ("bxi7adqn", "ott_qwen_grpo_eps_0_2.csv", "grpo", 0.2),
        ("2146seq9", "ott_qwen_grpo_eps_9_0.csv", "grpo", 9.0),
    ]
    provenance = []
    for run_id, csv_name, loss, epsilon in sources:
        paths = sorted((WORKSPACE / "wandb").rglob(f"run-{run_id}.wandb"))
        path = paths[0]
        config = {}
        history = {}
        run_name = ""
        for record in records(path):
            kind = record.WhichOneof("record_type")
            if kind == "run":
                run_name = record.run.display_name
            elif kind == "config":
                config.update(
                    {
                        item.key or ".".join(item.nested_key): json.loads(
                            item.value_json
                        )
                        for item in record.config.update
                    }
                )
            elif kind == "history":
                row = {
                    item.key or ".".join(item.nested_key): json.loads(item.value_json)
                    for item in record.history.item
                }
                if "train/global_step" in row:
                    history.setdefault(row["train/global_step"], {}).update(row)
        assert config["loss_type"] == loss, (run_id, config["loss_type"])
        assert config["clip_eps"] == epsilon, (run_id, config["clip_eps"])
        rows = csvrows(csv_name)
        for row in rows:
            logged = history[row["step"]]
            for csv_key, log_key in [
                ("f1_reward", "train/reward/f1_reward"),
                ("entropy", "train/entropy"),
            ]:
                assert abs(row[csv_key] - logged[log_key]) < 5e-7, (
                    run_id,
                    row["step"],
                    csv_key,
                )
        provenance.append(
            {
                "run_id": run_id,
                "name": run_name,
                "wandb_url": f"https://forge.coreweave.com/wandb/yssprashanth65536-student/huggingface/runs/{run_id}",
                "source": str(path.relative_to(WORKSPACE)),
                "csv_source": str((PAPER / "data" / csv_name).relative_to(WORKSPACE)),
                "export": f"assets/data/paper/{csv_name}",
                "verified_rows": len(rows),
                "config": {
                    key: config[key]
                    for key in [
                        "loss_type",
                        "clip_eps",
                        "mask_ratio_low",
                        "mask_ratio_high",
                    ]
                },
            }
        )
    (DATA / "clipping-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def export_qwen_neg_valley():
    source = (
        WORKSPACE / "verifiers/wandb/run-20261016_114651-mvw5wc16/run-mvw5wc16.wandb"
    )
    history = {}
    config = {}
    name = ""
    for record in records(source):
        kind = record.WhichOneof("record_type")
        if kind == "run":
            name = record.run.display_name
        elif kind == "config":
            config.update(
                {
                    item.key or ".".join(item.nested_key): json.loads(item.value_json)
                    for item in record.config.update
                }
            )
        elif kind == "history":
            row = {
                item.key or ".".join(item.nested_key): json.loads(item.value_json)
                for item in record.history.item
            }
            if "train/global_step" in row:
                history.setdefault(row["train/global_step"], {}).update(row)
    rows = [
        {"step": step, "reward": row["train/reward"], "entropy": row["train/entropy"]}
        for step, row in sorted(history.items())
        if "train/reward" in row and "train/entropy" in row
    ]
    assert rows and rows[-1]["step"] >= 60
    source_records = len(rows)
    rows = [row for row in rows if row["step"] <= 60]
    assert len(rows) == 60 and rows[-1]["step"] == 60
    with (DATA / "qwen-neg-valley-mvw5wc16.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "reward", "entropy"])
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "run_id": "mvw5wc16",
        "name": name,
        "source": str(source.relative_to(WORKSPACE)),
        "url": "https://forge.coreweave.com/wandb/yssprashanth65536-student/huggingface/runs/mvw5wc16",
        "records": len(rows),
        "source_records": source_records,
        "step_limit": 60,
        "smoothing_window": 1,
        "config": {
            key: value
            for key, value in config.items()
            if any(
                term in key
                for term in [
                    "loss_type",
                    "learning_rate",
                    "mask",
                    "valley",
                    "peak",
                    "advantage",
                ]
            )
        },
    }
    (DATA / "qwen-neg-valley-mvw5wc16.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return rows


def taxonomy_plots():
    selected_rows = export_qwen_neg_valley()
    for model in ["smol", "qwen"]:
        horizon = 60
        fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.4), layout="constrained")
        for suffix, label in [
            ("pos_favorite", "Pos-peak"),
            ("pos_challenger", "Pos-valley"),
            ("neg_favorite", "Neg-peak"),
            ("neg_challenger", "Neg-valley"),
        ]:
            for ax, metric in zip(axes, ["reward", "entropy"]):
                if model == "qwen" and suffix == "neg_challenger":
                    ax.plot(
                        [row["step"] for row in selected_rows],
                        [row[metric] for row in selected_rows],
                        color=COLORS[label],
                        label=label,
                        linewidth=1.9,
                        linestyle="--",
                    )
                    continue
                csvplot(
                    ax,
                    f"numina_{model}_{suffix}.csv",
                    label,
                    metric,
                    color=COLORS[label],
                    xmax=horizon,
                    linestyle="--" if "valley" in label else "-",
                )
        style(axes[0], "Training reward", "Which regimes learn?", horizon)
        style(axes[1], "Policy entropy (nats)", "Which way does entropy move?", horizon)
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="outside lower center",
            ncol=4,
            frameon=False,
        )
        finish(fig, f"quadrants-{model}")
        fig, raw_axes = plt.subplots(
            1, 1 if model == "smol" else 2, figsize=(10.3, 3.5), layout="constrained"
        )
        axes = np.atleast_1d(raw_axes)
        for suffix, label in (
            [
                ("pos", "Pos"),
                ("neg", "Neg"),
                ("favorite", "Peak"),
                ("challenger", "Valley"),
                ("dr_dapo-10-0.1", "DAPO"),
            ]
            if model == "smol"
            else [
                ("pos", "Pos"),
                ("neg", "Neg"),
                ("peak", "Peak"),
                ("valley", "Valley"),
                ("dr_dapo-10-0.1", "DAPO"),
            ]
        ):
            for ax, metric in zip(axes, ["reward", "entropy"]):
                csvplot(
                    ax,
                    f"numina_{model}_{suffix}.csv",
                    label,
                    metric,
                    window=1 if model == "qwen" else 10,
                    color=COLORS[label],
                    xmax=300,
                    linestyle="--" if label == "DAPO" else "-",
                )
        style(axes[0], "Training reward", "Combine two regimes", 300)
        if model == "qwen":
            style(axes[1], "Policy entropy (nats)", "The balance matters", 300)
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="outside lower center",
            ncol=5,
            frameon=False,
        )
        finish(fig, f"axes-{model}")


def benchmark_grid(panels, name, *, passk=False):
    """Keep all four datasets together, with one model per column."""
    datasets = ["OTT-QA", "HotpotQA", "MATH-500", "NuminaMath-LEAN"]
    models = ["Qwen3-4B", "SmolLM3-3B", "Gemma3-4B"]
    series_by_panel = {(p["dataset"], p["model"]): p["series"] for p in panels}
    assert len(series_by_panel) == len(panels) == len(datasets) * len(models)
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(11.5, 8.4),
        layout="constrained",
        sharex="row",
        sharey="row",
    )
    legend = {}
    for row, dataset in enumerate(datasets):
        for col, model in enumerate(models):
            ax = axes[row, col]
            for label, values in series_by_panel[dataset, model].items():
                xy = np.asarray(values)
                ax.plot(
                    xy[:, 0],
                    xy[:, 1],
                    color=COLORS[label],
                    label=label,
                    linewidth=2.4 if label == "WAPO" else 1.5,
                    marker="o",
                    markersize=3,
                )
            metric = (
                "Pass@k (%)"
                if passk
                else "Exact match (%)"
                if row < 2
                else "Accuracy (%)"
            )
            ax.set_ylabel(f"{dataset}\n{metric}" if col == 0 else "")
            if row == 0:
                ax.set_title(model, fontsize=12, pad=10)
            ax.grid(axis="y", color="#e7e9e6", linewidth=0.7)
            ax.set_axisbelow(True)
            if passk:
                ax.set_xscale("log", base=2)
                ax.set_xticks([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32])
                ax.set_xlim(0.9, 35)
            else:
                xmax = max(
                    xy[0]
                    for model_name in models
                    for values in series_by_panel[dataset, model_name].values()
                    for xy in values
                )
                ax.set_xlim(0, xmax)
                ax.set_xticks([0, 200, 400, 600] if xmax == 600 else [0, 250, 500])
            if row == len(datasets) - 1:
                ax.set_xlabel("Number of samples · k" if passk else "Training steps")
            handles, labels = ax.get_legend_handles_labels()
            legend.update(zip(labels, handles))
    ordered_labels = [
        label for label in ["Base", "DAPO", "GRPO", "GSPO", "WAPO"] if label in legend
    ]
    fig.legend(
        [legend[label] for label in ordered_labels],
        ordered_labels,
        loc="outside lower center",
        ncol=len(ordered_labels),
        frameon=False,
    )
    finish(fig, name)


def paper_plots():
    entropy_control_plot()
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.2), layout="constrained")
    for ax, values, title in zip(
        axes,
        [
            [0.78, 0.10, 0.05, 0.03, 0.039, 0.001],
            [0.35, 0.28, 0.22, 0.08, 0.069, 0.001],
        ],
        [
            "A concentrated distribution (Low entropy)",
            "A more spread-out distribution (High entropy)",
        ],
    ):
        p = np.array(values)
        boundary = float((p * p).sum())
        ax.bar(
            range(1, 7),
            p,
            color=[BLUE if v >= boundary else GOLD for v in p],
            width=0.65,
        )
        ax.axhline(boundary, color="#9a6e4b", linestyle="--", linewidth=1.2)
        ax.text(
            6.45,
            boundary + 0.025,
            f"C(p) = {boundary:.3f}",
            ha="right",
            color="#9a6e4b",
            fontsize=9,
        )
        ax.set(
            xlabel="Token",
            ylabel="Probability",
            title=title,
            ylim=(0, 0.88),
            xticks=range(1, 7),
        )
        ax.grid(axis="y", color="#e7e9e6")
        ax.set_axisbelow(True)
    finish(fig, "peak-valley")
    taxonomy_plots()
    clipping_plots()
    # These are the actual numerical summaries in peak_single_multi_illustration.tex.
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.3), layout="constrained")
    peak = np.array([58063, 61865, 60935])
    valley = np.array([7438, 3671, 2654])
    axes[0].bar(x - 0.18, 100 * peak / (peak + valley), 0.36, color=BLUE, label="Peak")
    axes[0].bar(
        x + 0.18, 100 * valley / (peak + valley), 0.36, color=GOLD, label="Valley"
    )
    axes[0].set(
        ylabel="Share of sampled tokens (%)",
        title="Peaks are sampled more often",
        ylim=(0, 105),
    )
    axes[1].bar(x - 0.18, [0.2719, 0.0966, 0.0698], 0.36, color=BLUE, label="Peak")
    axes[1].bar(x + 0.18, [1.0676, 0.8508, 0.7035], 0.36, color=GOLD, label="Valley")
    axes[1].set(
        ylabel="Mean context entropy (nats)",
        title="Valleys occur in less certain contexts",
    )
    for ax in axes:
        ax.set_xticks(x, ["SmolLM3", "Qwen3", "Gemma3"])
        ax.grid(axis="y", color="#e7e9e6")
        ax.set_axisbelow(True)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    finish(fig, "token-statistics")
    fig, ax = plt.subplots(figsize=(8.8, 3.5), layout="constrained")
    for suffix, label, c in [
        ("mask_neg_grp_new", "WAPO · (1 − q)", TEAL),
        ("mask_neg_seq_new", "Winner-normalized · (1 − q) / q", ORANGE),
        ("raftpp", "RAFT++", BLUE),
        ("psr", "PSR", GOLD),
    ]:
        csvplot(ax, f"numina_smol_{suffix}.csv", label, "reward", window=10, color=c)
    style(
        ax,
        "Training reward",
        "Positive-only design choices · SmolLM3 / NuminaMath-LEAN",
    )
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    finish(fig, "normalization")
    # Rebuild each paper panel directly from its explicit PGF coordinates.
    for source, name, datasets in [
        (
            "experiment_results_em.tex",
            "evaluation",
            ["OTT-QA", "HotpotQA", "MATH-500", "NuminaMath-LEAN"],
        ),
        ("hotpot_ott_passk.tex", "passk-qa", ["HotpotQA", "OTT-QA"]),
        ("math_numina_passk.tex", "passk-math", ["MATH-500", "NuminaMath-LEAN"]),
    ]:
        panels = re.findall(
            r"\\begin\{axis\}(.*?)\\end\{axis\}",
            (PAPER / source).read_text(),
            re.DOTALL,
        )
        assert len(panels) == len(datasets) * 3, (name, len(panels))
        structured = []
        for row, dataset in enumerate(datasets):
            for col, model in enumerate(["Qwen3-4B", "SmolLM3-3B", "Gemma3-4B"]):
                series = {}
                for key, coords in re.findall(
                    r"\\addplot\[wapo ([^\]]+) plot\]\s*coordinates\s*\{(.*?)\}",
                    panels[row * 3 + col],
                    re.DOTALL,
                ):
                    label = {
                        "drdapo": "DAPO",
                        "grpo": "GRPO",
                        "gspo": "GSPO",
                        "method": "WAPO",
                        "base model": "Base",
                    }[key]
                    xy = np.array(
                        [
                            (float(x), float(y))
                            for x, y in re.findall(
                                r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", coords
                            )
                        ]
                    )
                    series[label] = xy.tolist()
                structured.append(
                    {"dataset": dataset, "model": model, "series": series}
                )
        (DATA / f"{name}.json").write_text(json.dumps(structured, indent=2) + "\n")
    benchmark_grid(
        json.loads((DATA / "evaluation.json").read_text()), "evaluation-grid"
    )
    benchmark_grid(
        json.loads((DATA / "passk-qa.json").read_text())
        + json.loads((DATA / "passk-math.json").read_text()),
        "passk-grid",
        passk=True,
    )
    shutil.copytree(PAPER / "data", DATA / "paper", dirs_exist_ok=True)
    shutil.copy2(
        PAPER / "numina_smol_pos_favorite_step0_ir_vs_sampled_prob.pdf",
        DATA / "importance-ratio.pdf",
    )


# W&B's binary event stream uses 32 KiB LevelDB-style blocks.
def records(path):
    from wandb.proto.wandb_internal_pb2 import Record

    with path.open("rb") as f:
        if f.read(7) != b":W&B\xe1\xbe\x00":
            raise ValueError(f"Unknown W&B stream: {path}")
        fragment = b""
        while True:
            remaining = 32768 - f.tell() % 32768
            if remaining < 7:
                f.read(remaining)
                continue
            header = f.read(7)
            if len(header) < 7:
                return
            _, length, kind = struct.unpack("<IHB", header)
            if kind == 0:
                f.read(remaining - 7)
                continue
            data = f.read(length)
            if len(data) != length:
                return  # interrupted stream: retain complete records
            if kind == 1:
                whole = data
            elif kind == 2:
                fragment = data
                continue
            elif kind == 3:
                fragment += data
                continue
            elif kind == 4:
                whole = fragment + data
                fragment = b""
            else:
                raise ValueError((path, f.tell(), kind))
            record = Record()
            record.ParseFromString(whole)
            yield record


def is_relevant(name, date):
    n = name.lower()
    return bool(
        re.search(
            r"(favorite|challenger|peak|valley|neg[-_]val(?:[-_]|$)|pos[-_]val(?:[-_]|$)|pos_no_val)",
            n,
        )
        or re.search(r"(numina|math).*[-_](?:pos|neg)(?:-norm|-on-policy)?$", n)
        or (
            date.startswith("2026-02")
            and "qwen" in n
            and ("mask-neg" in n or n == "ott-sft-format-50-bf16-qwen4b-seqir")
        )
    )


def extract_runs():
    seen = set()
    runs = []
    for root in [WORKSPACE / "wandb", WORKSPACE / "verifiers/wandb"]:
        for path in sorted(root.rglob("run-*.wandb")):
            rid = path.stem[4:]
            if rid in seen:
                continue
            seen.add(rid)
            stream = records(path)
            for rec in stream:
                if rec.WhichOneof("record_type") == "run":
                    break
            else:
                continue
            name = rec.run.display_name
            date = datetime.fromtimestamp(rec.run.start_time.seconds, UTC).strftime(
                "%Y-%m-%d"
            )
            if not is_relevant(name, date):
                continue
            config = {
                h.key or ".".join(h.nested_key): json.loads(h.value_json)
                for h in rec.run.config.update
            }
            history = []
            for rec in stream:
                kind = rec.WhichOneof("record_type")
                if kind == "config":
                    config.update(
                        {
                            h.key or ".".join(h.nested_key): json.loads(h.value_json)
                            for h in rec.config.update
                        }
                    )
                elif kind == "history":
                    row = {
                        h.key or ".".join(h.nested_key): json.loads(h.value_json)
                        for h in rec.history.item
                    }
                    history.append(
                        {
                            k: round(v, 8) if isinstance(v, float) else v
                            for k, v in row.items()
                            if isinstance(v, (int, float))
                            and np.isfinite(v)
                            and (k.startswith("train/") or k == "_step")
                            and not any(t in k for t in ["timing/", "wall_clock/"])
                        }
                    )
            model = next(
                (m for m in ["qwen", "smol", "gemma", "llama"] if m in name.lower()),
                "other",
            )
            dataset = (
                "NuminaMath-LEAN"
                if "numina" in name.lower()
                else "MATH-500"
                if name.startswith("math")
                else "OTT-QA"
                if name.startswith("ott")
                else "HotpotQA"
                if name.startswith("hotpot")
                else "Other"
            )
            selected_config = {
                k: v
                for k, v in config.items()
                if any(
                    t in k
                    for t in [
                        "mask",
                        "clip",
                        "learning_rate",
                        "loss_type",
                        "advantage",
                        "peak",
                        "valley",
                        "favorite",
                        "challenger",
                    ]
                )
            }
            columns = {
                key: [
                    [row.get("train/global_step", row.get("_step", 0)), row[key]]
                    for row in history
                    if key in row
                ]
                for key in sorted(
                    {
                        k
                        for row in history
                        for k in row
                        if k not in ["_step", "train/global_step", "train/epoch"]
                    }
                )
            }
            item = {
                "id": rid,
                "name": name,
                "date": date,
                "model": model,
                "dataset": dataset,
                "steps": max(
                    (r.get("train/global_step", r.get("_step", 0)) for r in history),
                    default=0,
                ),
                "records": len(history),
                "source": str(path.relative_to(WORKSPACE)),
                "url": f"https://forge.coreweave.com/wandb/yssprashanth65536-student/huggingface/runs/{rid}",
                "config": selected_config,
            }
            runs.append(item)
            (DATA / "runs" / f"{rid}.json").write_text(
                json.dumps(dict(**item, metrics=columns), separators=(",", ":")) + "\n"
            )
            if history:
                fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.8), layout="constrained")
                reward = (
                    "train/reward/f1_reward"
                    if dataset in ["OTT-QA", "HotpotQA"]
                    and "train/reward/f1_reward" in columns
                    else "train/reward"
                )
                for ax, metric, title in [
                    (
                        axes[0],
                        reward,
                        "F1 reward" if "f1" in reward else "Training reward",
                    ),
                    (axes[1], "train/entropy", "Policy entropy (nats)"),
                ]:
                    points = columns.get(metric, [])
                    if points:
                        xy = np.array(points)
                        ax.plot(
                            xy[:, 0], xy[:, 1], alpha=0.19, color=TEAL, linewidth=0.6
                        )
                        ax.plot(
                            xy[:, 0], smooth(xy[:, 1], 10), color=TEAL, linewidth=1.4
                        )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "Metric not recorded",
                            ha="center",
                            transform=ax.transAxes,
                        )
                    style(ax, title)
                finish(fig, "run-" + rid)
    (DATA / "run-manifest.json").write_text(json.dumps(runs, indent=2) + "\n")
    print(
        "Exported",
        len(runs),
        "candidate runs;",
        sum(r["records"] > 0 for r in runs),
        "with history",
    )
    return runs


def snapshot_statistics():
    out = []
    names = [
        "numina_lean_math-smol3b-256_64_8-drdapo-pos_peak",
        "numina_lean_math-qwen4b-256-64-8-drdapo",
        "numina-gemma-256-64-8-drdapo-neg_val-ent_thresh_neg01",
    ]
    for name in names:
        path = WORKSPACE / "rl_logs/plots" / name / "data/scatter_step_0.npz"
        d = np.load(path)
        p = d["p_s"]
        c = d["C_p"]
        valid = np.isfinite(p) & np.isfinite(c)
        peaks = int((p[valid] >= c[valid]).sum())
        n = int(valid.sum())
        out.append(
            {
                "run": name,
                "source": str(path.relative_to(WORKSPACE)),
                "step": 0,
                "tokens": n,
                "peaks": peaks,
                "valleys": n - peaks,
                "peak_percent": 100 * peaks / n,
            }
        )
    (DATA / "token-snapshots.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    (DATA / "runs").mkdir(parents=True, exist_ok=True)
    paper_plots()
    verify_clipping_sources()
    extract_runs()
    snapshot_statistics()
