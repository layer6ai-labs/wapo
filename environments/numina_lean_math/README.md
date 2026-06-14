# numina-lean-math

Single-turn math environment built from `AI-MO/NuminaMath-LEAN`.

## Data filter

- Keeps only rows with `question_type` in:
  - `math-word-problem`
  - `MCQ`
- Drops rows with missing/empty `problem` or `answer`.
- Keeps only simple numeric answers (`int`, `decimal`, or `fraction`), and drops non-numeric labels such as `unknown`, symbolic forms, intervals, or multi-answer strings.

## Prompt format

Each prompt asks the model to respond as:

```
<think>...</think>
Answer: <final answer>
```

Reward is exact match on parsed `Answer: ...`, and format is required.

## Train/validation split

- The filtered dataset is shuffled with `seed`.
- First `eval_size` examples are used as validation (`eval_dataset`).
- Remaining examples are used for training (`dataset`).
- Defaults: `eval_size=1000`, `seed=42`.

## Usage

```python
import verifiers as vf

env = vf.load_environment("numina-lean-math", eval_size=1000, seed=42)
```

CLI:

```bash
vf-install numina-lean-math --from-repo
vf-eval numina-lean-math -n 50
```
