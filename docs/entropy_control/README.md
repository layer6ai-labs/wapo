# Entropy dynamics research post

A separate, buildless blog post in the WAPO repository. Its HTML, CSS, JavaScript, figures, manuscript PDF, and CSV snapshots live in this folder. The existing `docs/wapo_page/` is unchanged. No remote service or JavaScript dependency is needed to read the post.

From the repository root:

```bash
python3 -m http.server 8000 --directory docs
```

Read `http://localhost:8000/entropy_control/`. When this repository's `docs` folder is published through GitHub Pages, the corresponding path is `/wapo/entropy_control/`. This change does not modify publication settings or deploy the page.

The visual language follows the WAPO research page: green palette, serif headings, reading sidebar, expandable evidence, and a dark-mode switch. Scientific plots have separate narrow-screen layouts. All paper links open the supplied `neurips_2026/rlvr_entropy_control.pdf`, copied to `assets/paper.pdf`.

## Figures and sources

`assets/data/` contains the exact CSV snapshots from `neurips_2026/overleaf/plots/`. `provenance.json` records their paths and SHA-256 hashes. All figures show the supplied coordinates without additional smoothing or resampling. The main threshold figure retains the manuscript's axis ranges and truncated trajectories.

The four plot groups are:

- `entropy-filtering-reward-entropy`: the four main one-sided/two-sided comparisons.
- `entropy-first-order-collapse`: baseline reward, measured entropy, and accumulated first-order estimates.
- `entropy-80-20-rule`: highest-entropy 20% selection versus baseline.
- `peak-valley-entropy`: peak-only versus valley-only training.

Rebuild the SVGs with a Python environment containing Matplotlib and NumPy:

```bash
MPLCONFIGDIR=/tmp/entropy-control-mpl python tools/build_assets.py
```

Run that command from this folder, or supply the script's full path from another directory. To refresh both CSV snapshots and the PDF from the original source:

```bash
MPLCONFIGDIR=/tmp/entropy-control-mpl python tools/build_assets.py \
  --source-root /home/prasanth/research/neurips_2026/overleaf
```

## Editorial scope

Claims concern the reported entropy-growth failures and plotted horizons. The prose distinguishes the local logit-space estimate from a full-network prediction, explains that the normal sampled-token importance ratio remains, and makes no measured runtime or memory claim.

The requested X status could not be retrieved directly. It is retained as related discussion, without quoting it or attributing specific findings to it. The KL paragraph instead cites the accessible primary paper *A Comedy of Estimators* and Bartoldson's own EMA-reference analysis. Full-distribution KL and sampled log-ratio approximations have different storage requirements; the post preserves that distinction.
