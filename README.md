<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# Winner Advantage Policy Optimization

<div align="center">

[![arxiv](https://img.shields.io/static/v1?label=arXiv&message=2606.16154&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2606.16154)

</div>

## Verifiers (fork)

This is a fork of [PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers) extending `vf.RLTrainer` with research loss functions. See the original repo for installation, environment documentation, and general usage.

## Quick Installation

```bash
cd verifiers
uv sync --all-extras && uv pip install flash-attn --no-build-isolation
uv run pre-commit install
source .venv/bin/activate
```

## Loss Functions

### Baseline loss functions

These are inherited from the original repo and commonly used as baselines.

| `loss_type` | Importance ratio | Advantage | Normalization |
|---|---|---|---|
| `grpo` | token IR, symmetric clip `[1−ε, 1+ε]` | (r − mean) / std | per-sequence token mean |
| `gspo` | sequence IR, symmetric clip `[1−ε, 1+ε]` | (r − mean) / std | per-sequence token mean |
| `dr_dapo` | token IR, asymmetric clip `[low, high]` | r − mean | per-group token mean |

- **`grpo`** clips the per-token importance ratio symmetrically and normalizes the advantage by the within-group standard deviation. The loss is averaged per sequence and then summed across sequences.
- **`gspo`** is like GRPO but uses a sequence-level importance ratio (product of per-token ratios) instead of per-token ratios. This makes the update more conservative: a sequence is only reinforced if the policy has moved in the right direction across all tokens jointly.
- **`dr_dapo`** uses an asymmetric clip range (`mask_ratio_low` / `mask_ratio_high`) and drops std normalization. Advantage is `r − group_mean`. Normalization is per group token count.

### New loss functions (this fork)

`wapo` trains exclusively on positive rollouts (sequences with above-average reward) and uses importance sampling with a one-sided upper clip.

| `loss_type` | Advantage | Group denominator |
|---|---|---|
| `wapo` | r − mean | group\_size × max\_seq\_len |

- **`wapo`** normalizes by the full group size, so a group with fewer positives contributes proportionally less.

Set `loss_type` and `clip_eps` in your training config:

```toml
[trainer.args]
loss_type = "wapo"
clip_eps = 9.0
rollouts_per_example = 8
```

To normalize by actual kept token counts instead:

```toml
normalize_by_kept_tokens = true
```

## Peak and Valley Tokens

Each completion token is classified into one of two bins based on how the current policy's probability `p_train` compares to the collision probability `Σ p_i²` (the second moment of the vocabulary distribution at that position):

| Bin | Condition | Interpretation |
|---|---|---|
| `peak` | `p_train ≥ Σ p_i²` | The model places high mass on this token — it is a confident, low-entropy choice |
| `valley` | `p_train < Σ p_i²` | The model's probability is below the collision threshold — the token comes from a flat, high-entropy region of the distribution |

`Σ p_i²` is the expected probability of a token drawn from the same distribution, so the threshold separates tokens the model is already "committed to" from tokens it is still uncertain about. Reinforcing valleys pushes the model to commit to specific tokens in high-entropy positions; reinforcing peaks pushes it to reinforce already-confident choices.

Bins are further split by the sign of the sequence advantage, giving four groups:

| Full bin name | Meaning |
|---|---|
| `pos_peak` | Confident tokens in above-average rollouts |
| `pos_valley` | Uncertain tokens in above-average rollouts |
| `neg_peak` | Confident tokens in below-average rollouts |
| `neg_valley` | Uncertain tokens in below-average rollouts |

### Filtering by bin

Use `include_bins` to train only on specific bins, or `exclude_bins` to drop specific bins. Both accept any mix of full names (`pos_peak`), type shorthands (`peak` → both `pos_peak` and `neg_peak`), and sign shorthands (`pos` → both `pos_peak` and `pos_valley`).

```toml
[trainer.args]
# Train only on positive-advantage tokens, keeping both peak and valley:
include_bins = ["pos"]

# Or exclude negative-advantage valleys (uncertain tokens in bad rollouts):
exclude_bins = ["neg_valley"]
```

Bin counts (eligible and kept) are logged to W&B at every step for diagnostics.

## Example Configs

Ready-to-run configs are in [`configs/vf-rl/`](configs/vf-rl/). Run with:

```bash
uv run vf-rl @ configs/vf-rl/<config>.toml
```

| Config | Environment | Notes |
|---|---|---|
| [`numina_lean_math.toml`](configs/vf-rl/numina_lean_math.toml) | `numina_lean_math` | Dataset loaded from HF automatically |
| [`math_prm.toml`](configs/vf-rl/math_prm.toml) | `math_prm` | Set `train_path` / `test_path` in `[env.args]` |
| [`ott_qa.toml`](configs/vf-rl/ott_qa.toml) | `vf-rag-agent` | Set dataset paths and `retrieve_url` in `[env.args]` |
| [`hotpot_qa.toml`](configs/vf-rl/hotpot_qa.toml) | `vf-rag-agent` | Set dataset paths and `retrieve_url` in `[env.args]` |

The RAG-agent configs (`ott_qa`, `hotpot_qa`) require a running retrieval server; point `retrieve_url` at it.

## Retrieval Server

The RAG agent calls an HTTP retrieval server at the URL set by `retrieve_url`. We use the server from [Search-R1](https://github.com/PeterGriffinJin/Search-R1), which indexes a Wikipedia corpus with FAISS and serves dense retrieval via a FastAPI endpoint.

### Corpus download

**OTT-QA** (tables + linked passages from Wikipedia):

```bash
mkdir -p data/ott-corpus && cd data/ott-corpus
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_plain_tables.json
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_passages.json
cd ../..
```

**Wikipedia (for HotpotQA / general use)** — download the pre-built FAISS index and corpus from Search-R1:

```bash
save_path=data/wiki
python scripts/download.py --save_path $save_path          # from Search-R1 repo
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

See the [OTT-QA repo](https://github.com/wenhuchen/OTT-QA) and [Search-R1 repo](https://github.com/PeterGriffinJin/Search-R1) for full corpus preparation details.

### Server setup

```bash
conda create -n retriever python=3.10
conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini uvicorn fastapi
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

Launch (from the Search-R1 repo):

```bash
python search_r1/search/retrieval_server.py \
    --index_path data/wiki/e5_Flat.index \
    --corpus_path data/wiki/wiki-18.jsonl \
    --model intfloat/e5-base-v2 \
    --topk 3 \
    --faiss_gpu
```

Then set `retrieve_url = "http://<host>:<port>/retrieve"` in your config.

### API contract

The environment sends a `POST /retrieve` request and expects a response that is a JSON array of per-query result lists, where each result is a plain text string:

**Request:**
```json
{"queries": ["your question here"], "top_k": 3}
```

**Response:**
```json
[
  [
    "retrieved text ...",
    "retrieved text ...",
    "retrieved text ..."
  ]
]
```

The outer list has one entry per query.

## Datasets

Preprocessed datasets used in our runs are available from a shared Google Drive folder. Download them with [`gdown`](https://github.com/wkentaro/gdown):

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1ZJ5nR5AZ3iAB4wNJFJJR_LLe8IHb29KF" -O data/
```

This creates the following layout under `data/`:

```
data/
  ott-qa/
    mix-10000-v1.json       # OTT-QA train
    ott-eval.json           # OTT-QA eval
  hotpot/
    hotpot_dev_fullwiki_v1.json   # HotpotQA eval
  hotpot_train_v1.1_rl.json       # HotpotQA train
  math_splits/
    train.jsonl             # Math train
    test.jsonl              # Math test
```

The example configs in `configs/vf-rl/` already reference these paths under `data/`. Update `[env.args]` in each config if you place the data elsewhere.

## Citing

If you use any part of this repository in your research, please cite the associated paper with the following bibtex entry:

```
@article{yss2026wapo,
  title={A Gradient Perspective on RLVR Stability and Winner Advantage Policy Optimization},
  author={Prasanth YSS and Zhichen Ren and Rasa Hosseinzadeh and Ilan Gofman and Yuqi Chen and Zhaoyan Liu and Guangwei Yu and Jesse C. Cresswell and Satya Krishna Gorti},
  journal={arXiv:2606.16154},
  year={2024}
}
```

## License

This data and code is licensed under the MIT License, copyright by Layer 6 AI.
