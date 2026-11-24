"""Rebuild the blog's scientific figures from its manuscript CSV snapshots."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

PAGE = Path(__file__).resolve().parents[1]
DATA = PAGE / "assets" / "data"
FIGURES = PAGE / "assets" / "figures"
GROUPS = (
    "entropy-filtering-reward-entropy",
    "entropy-first-order-collapse",
    "entropy-80-20-rule",
    "peak-valley-entropy",
    "ott-qwen-threshold-seeds",
)
BLUE, ORANGE, RED, GREEN = "#0072B2", "#BD7A00", "#D55E00", "#008568"
METRICS = ("reward", "entropy", "cumulative_first_order_entropy_change")
TITLES = ("Reward", "Trainer entropy", r"Accumulated predicted $\widehat{\Delta H}$")


def import_sources(source: Path) -> None:
    manifest: dict[str, object] = {
        "paper": "Diagnosing and Stabilizing RLVR Training through Entropy Dynamics",
        "source_root": "neurips_2026/overleaf",
        "transform": "CSV coordinates as supplied; no additional smoothing or resampling",
        "thresholds": {"ott-qwen": 0.1, "other_settings": 0.2, "lower": -0.1},
        "learning_rate_omitted_for_filtering": True,
        "files": [],
    }
    records = []
    for group in GROUPS:
        target = DATA / group
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted((source / "plots" / group / "data").glob("*.csv")):
            shutil.copyfile(path, target / path.name)
            records.append(
                {
                    "path": f"{group}/{path.name}",
                    "source": f"plots/{group}/data/{path.name}",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    assert records, "No manuscript CSVs found"
    manifest["files"] = records
    (DATA / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(
        source.parent / "rlvr_entropy_control.pdf", PAGE / "assets" / "paper.pdf"
    )


def read(group: str, name: str) -> np.ndarray:
    return np.genfromtxt(DATA / group / f"{name}.csv", delimiter=",", names=True)


def style(ax: plt.Axes, title: str, xmax: int) -> None:
    ax.set_title(title, loc="left", pad=10, fontweight="normal")
    ax.set_xlim(0, xmax)
    ax.grid(axis="both", color="#dce7e1", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b2c4ba")
    ax.tick_params(length=0, pad=6)
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel("Gradient steps")


def save(fig: plt.Figure, name: str) -> None:
    path = FIGURES / f"{name}.svg"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def thresholds(mobile: bool) -> None:
    settings = (
        ("ott-qwen", "Qwen3-4B · OTT-QA", 400, 0.65),
        ("numina-qwen", "Qwen3-4B · NuminaMath-LEAN", 400, 0.7),
        ("math-gemma", "Gemma3-4B · MATH-500", 1000, 0.9),
        ("numina-gemma", "Gemma3-4B · NuminaMath-LEAN", 600, 0.62),
    )
    fig, axes = plt.subplots(
        4 if mobile else 2,
        2 if mobile else 4,
        figsize=(7.5, 13.8) if mobile else (14, 7.5),
    )
    fig.subplots_adjust(
        hspace=0.9,
        wspace=0.38,
        top=0.92 if mobile else 0.88,
        bottom=0.06,
    )
    for i, (name, label, xmax, reward_max) in enumerate(settings):
        row, col = (i, 0) if mobile else (i // 2, (i % 2) * 2)
        for offset, metric in enumerate(METRICS[:2]):
            ax = axes[row, col + offset]
            for suffix, color, line in (
                ("baseline", BLUE, "-"),
                ("upper", ORANGE, "-"),
                ("two-sided", RED, "--"),
            ):
                series = read(GROUPS[0], f"{name}-{suffix}")
                ax.plot(
                    series["gradient_step"],
                    series[metric],
                    color=color,
                    linewidth=1.6,
                    linestyle=line,
                )
            style(ax, TITLES[offset], xmax)
            ax.set_ylim(0, reward_max if offset == 0 else 13)
        left, right = axes[row, col].get_position(), axes[row, col + 1].get_position()
        fig.text(
            (left.x0 + right.x1) / 2,
            left.y1 + (0.035 if mobile else 0.075),
            label,
            ha="center",
            va="bottom",
            fontsize=12 if mobile else 11,
            linespacing=1.5,
        )
    legend(
        fig,
        ("Baseline", "One-sided filter", "Two-sided filter"),
        (BLUE, ORANGE, RED),
        ("-", "-", "--"),
        mobile,
    )
    save(fig, "thresholds-mobile" if mobile else "thresholds")


def legend(
    fig: plt.Figure,
    labels: tuple[str, ...],
    colors: tuple[str, ...],
    lines: tuple[str, ...],
    mobile: bool,
    columns: int | None = None,
) -> None:
    handles = [
        Line2D([0], [0], color=color, lw=2, linestyle=line)
        for color, line in zip(colors, lines, strict=True)
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        frameon=False,
        ncol=1 if mobile else (columns or len(labels)),
        fontsize=16,
    )


def dynamics(mobile: bool) -> None:
    runs = (
        ("qwen-ott", "OTT-QA", BLUE),
        ("qwen-numina", "NuminaMath-LEAN", RED),
        ("gemma-math", "MATH-500", GREEN),
        ("gemma-numina", "NuminaMath-LEAN", "#A15B8A"),
    )
    fig = plt.figure(figsize=(7, 19) if mobile else (14, 8.7))
    grid = fig.add_gridspec(
        8 if mobile else 4,
        1 if mobile else 3,
        height_ratios=(1, 1, 1, 0.22) * 2 if mobile else (1, 0.22) * 2,
        hspace=0.85 if mobile else 0.65,
        wspace=0.35,
        left=0.09,
        right=0.98,
        top=0.94,
        bottom=0.04,
    )
    for model in range(2):
        for j, metric in enumerate(METRICS):
            ax = fig.add_subplot(
                grid[model * 4 + j, 0] if mobile else grid[model * 2, j]
            )
            for name, label, color in runs[model * 2 : model * 2 + 2]:
                series = read(GROUPS[1], name)
                ax.plot(
                    series["gradient_step"],
                    series[metric],
                    color=color,
                    linewidth=1.6,
                    label=label,
                )
            title = f"{'Qwen3-4B' if model == 0 else 'Gemma3-4B'}\n{TITLES[j]}"
            style(ax, title, 400 if model == 0 else 800)
            limits = (
                ((0, 0.65), (0, 13), (-0.035, 0.065))
                if model == 0
                else ((0, 0.85), (0, 13), (-0.001, 0.0105))
            )
            ax.set_ylim(*limits[j])
        legend_ax = fig.add_subplot(
            grid[model * 4 + 3, :] if mobile else grid[model * 2 + 1, :]
        )
        legend_ax.set_axis_off()
        model_runs = runs[model * 2 : model * 2 + 2]
        handles = [Line2D([0], [0], color=color, lw=2) for _, _, color in model_runs]
        legend_ax.legend(
            handles,
            [label for _, label, _ in model_runs],
            loc="center",
            ncol=2,
            frameon=False,
            fontsize=16,
            columnspacing=2,
            handlelength=2.4,
        )
    save(fig, "dynamics-mobile" if mobile else "dynamics")


def methods(
    group: str,
    configurations: tuple[tuple[str, str, int], ...],
    labels: tuple[str, ...],
    suffixes: tuple[str, ...],
    name: str,
    mobile: bool,
) -> None:
    rows = len(configurations)
    fig, axes = (
        plt.subplots(rows * 3, 1, figsize=(7, rows * 8.5), squeeze=False)
        if mobile
        else plt.subplots(rows, 3, figsize=(14, rows * 3.5), squeeze=False)
    )
    top = 0.94 if mobile else 0.88
    fig.subplots_adjust(hspace=0.7, wspace=0.36, top=top, bottom=0.09)
    for i, (prefix, label, xmax) in enumerate(configurations):
        for j, metric in enumerate(METRICS):
            ax = axes[i * 3 + j, 0] if mobile else axes[i, j]
            for suffix, color in zip(suffixes, (BLUE, RED), strict=True):
                series = read(group, f"{prefix}{suffix}")
                ax.plot(
                    series["gradient_step"], series[metric], color=color, linewidth=1.6
                )
            style(ax, f"{label}\n{TITLES[j]}", xmax)
    legend(fig, labels, (BLUE, RED), ("-", "-"), mobile)
    save(fig, f"{name}-mobile" if mobile else name)


def multi_seed(seed: int, mobile: bool) -> None:
    variants = [
        ("baseline", "Baseline", BLUE, "-"),
        ("01", r"$\widehat{\Delta H}<0.1$", ORANGE, "-"),
    ]
    if seed == 42:
        variants.extend(
            [
                ("02", r"$\widehat{\Delta H}<0.2$", "#56B4E9", "-"),
                ("02-01", r"$-0.2<\widehat{\Delta H}<0.1$", GREEN, "--"),
            ]
        )
    variants.append(("01-01", r"$-0.1<\widehat{\Delta H}<0.1$", RED, "--"))
    fig, axes = (
        plt.subplots(3, 1, figsize=(7, 10))
        if mobile
        else plt.subplots(1, 3, figsize=(14, 4.5))
    )
    fig.subplots_adjust(hspace=0.7, wspace=0.35, top=0.88, bottom=0.15)
    limits = ((0, 0.72), (0, 12), (-0.55, 0.1))
    ticks = ((0, 0.2, 0.4, 0.6), (0, 3, 6, 9, 12), (-0.5, -0.3, -0.1, 0.1))
    for j, (ax, metric) in enumerate(zip(axes, METRICS, strict=True)):
        for suffix, _, color, line in variants:
            series = read(GROUPS[4], f"seed{seed}-{suffix}")
            ax.plot(
                series["gradient_step"],
                series[metric],
                color=color,
                linestyle=line,
                linewidth=1.6,
            )
        style(ax, f"Seed {seed}\n{TITLES[j]}", 1400)
        ax.set_xticks((0, 350, 700, 1050, 1400))
        ax.set_ylim(*limits[j])
        ax.set_yticks(ticks[j])
    legend(
        fig,
        tuple(variant[1] for variant in variants),
        tuple(variant[2] for variant in variants),
        tuple(variant[3] for variant in variants),
        mobile,
        columns=3,
    )
    save(fig, f"ott-qwen-seed{seed}{'-mobile' if mobile else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Import fresh CSVs and PDF from the Overleaf directory",
    )
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.source_root is not None:
        import_sources(args.source_root)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": "#18332b",
            "axes.labelcolor": "#52675c",
            "xtick.color": "#52675c",
            "ytick.color": "#52675c",
            "svg.fonttype": "none",
            "svg.hashsalt": "entropy-control",
        }
    )
    for mobile in (False, True):
        thresholds(mobile)
        dynamics(mobile)
        for seed in (42, 43, 44):
            multi_seed(seed, mobile)
        methods(
            GROUPS[2],
            (("numina-qwen-", "Qwen3-4B", 600), ("numina-gemma-", "Gemma3-4B", 750)),
            ("Baseline", "Highest-entropy 20%"),
            ("baseline", "80-20"),
            "high-entropy-tokens",
            mobile,
        )
        methods(
            GROUPS[3],
            (("", "SmolLM3-3B · NuminaMath-LEAN", 800),),
            ("Peak only", "Valley only"),
            ("peak", "valley"),
            "peak-valley",
            mobile,
        )


if __name__ == "__main__":
    main()
