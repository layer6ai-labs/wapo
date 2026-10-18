# WAPO research page

A buildless GitHub Pages site. Open `index.html` to read the page and use the local gradient illustration. No account, API key, JavaScript framework, external font service, or live W&B connection is required.

For a local HTTP preview, run from the `wapo` repository:

```bash
python3 -m http.server 8000 --directory docs/wapo_page
```

The page is then at `http://localhost:8000`. With GitHub Pages publishing the repository’s `docs` folder, its path is `/wapo/wapo_page/`. Publication settings are not changed by this directory.

A brief overview maps the argument, followed by the original introduction under Background. The main narrative then goes from the clipping puzzle to local gradient analysis and four token regimes, returns to clipping, tests direct entropy control, and examines positive-only training. All benchmark results are displayed on the main page. Token-selection details and review questions use native `<details>` elements. The former experiment-browser section has been removed; its existing source-data artifacts are retained.

The header includes a dark-mode switch. It follows the system preference until the reader chooses a mode, then remembers that choice in browser storage. The switch also works for the current visit when storage is unavailable. `theme.js` applies the initial theme before the stylesheet loads.

## Sources and rebuilding

`tools/build_assets.py` rebuilds figures and the local evidence bundle. Run it from the surrounding research workspace with Python, NumPy, Matplotlib, and W&B installed:

```bash
MPLCONFIGDIR=/tmp/wapo-mpl verifiers/.venv/bin/python wapo/docs/wapo_page/tools/build_assets.py
```

Inputs are `EMNLP_2026/wapo_paper/plots`, `neurips_2026/rlvr_entropy_control_overleaf/plots/entropy-filtering-reward-entropy`, `wandb`, `verifiers/wandb`, and the selected `rl_logs/plots` token snapshots. The builder does not rewrite page content. It excludes system telemetry and console output from the exported run bundle and needs no network access.

- Benchmark plots use CSVs or explicit evaluation coordinates from the supplied manuscript. Numeric checkpoint and pass@k values are exported as JSON.
- The Qwen negative-valley curve uses **the first 60 steps** of `verifiers/wandb/run-20261016_114651-mvw5wc16/run-mvw5wc16.wandb`. Its raw values and provenance are in `assets/data/qwen-neg-valley-mvw5wc16.csv` and `.json`.
- Qwen taxonomy, combined-mask, and clipping curves use raw recorded values. Captions identify smoothing on the other training plots.
- Gemma/DAPO and Qwen/GRPO clipping comparisons are checked row by row against their named W&B histories. Configuration and source paths are recorded in `assets/data/clipping-provenance.json`.
- The direct entropy-control comparison uses the OTT-Qwen and Numina-Gemma panels from Figure 2 of `neurips_2026/131_Diagnosing_and_Stabilizing.pdf`, rebuilt from the supplied plot CSVs without smoothing. `assets/data/entropy-control.json` records the displayed values, run IDs, thresholds, source paths, and source-file hashes.
- The peak/valley illustration uses normalized toy distributions. It is explicitly labeled as an illustration.
- The importance-ratio panel is the original scientific figure, rendered from PDF with `pdftoppm`.
- Additional review controls are summarized in the questions section. Main benchmark results and those additional controls remain distinguishable.

To regenerate the ratio image:

```bash
pdftoppm -scale-to 1250 -png -singlefile \
  EMNLP_2026/wapo_paper/plots/numina_smol_pos_favorite_step0_ir_vs_sampled_prob.pdf \
  wapo/docs/wapo_page/assets/figures/importance-ratio
```
