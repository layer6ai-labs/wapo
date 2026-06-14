import re

from datasets import load_dataset

import verifiers as vf

_FORMAT_RE = re.compile(r"^\s*<think>(.+?)</think>\s*Answer:\s*(.+?)\s*$", re.DOTALL)


def extract_answer(text: str) -> str:
    """Extract answer from '<think>...</think> Answer: $num' or '\\boxed{...}' format."""
    m = _FORMAT_RE.match(text)
    if m:
        return m.group(2).strip()
    boxed = _extract_boxed(text)
    if boxed:
        return boxed
    match = re.search(r"Answer:\s*(.+)", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _extract_boxed(text: str) -> str:
    """Extract content of the last \\boxed{...}, handling nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return ""
    depth = 0
    start = idx + len(r"\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return ""


def _check_format(text: str, require_boxed: bool) -> bool:
    if require_boxed:
        return bool(_extract_boxed(text))
    return _FORMAT_RE.match(text) is not None


_BOXED_MODELS = {
    "Qwen/Qwen2.5-Math-7B",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "Qwen/Qwen2.5-Math-1.5B",
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "Qwen/Qwen2.5-Math-72B",
    "Qwen/Qwen2.5-Math-72B-Instruct",
}


def _is_boxed_model(model: str | None) -> bool:
    if model is None:
        return False
    return any(m in model for m in _BOXED_MODELS)


def load_environment(
    n: int = 100000,
    train_path: str = "/home/zhichen/math_splits/train.jsonl",
    test_path: str = "/home/zhichen/math_splits/test.jsonl",
    split: str = "train",
    model: str | None = None,
    **kwargs,
) -> vf.Environment:
    boxed_format = _is_boxed_model(model)
    parser = vf.Parser(extract_fn=extract_answer)

    def reward_func(parser, completion, answer, **kwargs):
        """Reward 1.0 only if format is correct AND answer matches exactly."""
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        else:
            text = completion
        if not _check_format(text, require_boxed=boxed_format):
            return 0.0
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    rubric = vf.Rubric(
        parser=parser,
        funcs=[reward_func],
        weights=[1.0],
    )

    path = train_path if split == "train" else test_path
    ds = load_dataset("json", data_files=path, split="train")

    def preprocess(example):
        problem = example["problem"]
        if boxed_format:
            prompt = (
                problem.rstrip()
                + "\n\nSolve the problem and put your final answer inside \\boxed{}."
            )
        else:
            prompt = (
                problem.rstrip()
                + "\n\nPut your reasoning inside <think>...</think> tags, "
                "then write your final answer as: Answer: <your answer>"
            )
        return {
            "prompt": [{"role": "user", "content": prompt}],
            "answer": example["answer"],
        }

    ds = ds.map(preprocess)
    if n > 0 and n < len(ds):
        ds = ds.shuffle(seed=42).select(range(n))

    env = vf.SingleTurnEnv(
        dataset=ds,
        parser=parser,
        rubric=rubric,
    )
    return env
