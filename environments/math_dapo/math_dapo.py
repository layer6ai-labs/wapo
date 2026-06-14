import re

from datasets import load_dataset

import verifiers as vf

_FORMAT_RE = re.compile(r"^\s*<think>(.+?)</think>\s*Answer:\s*(.+?)\s*$", re.DOTALL)


def extract_answer(text: str) -> str:
    """Extract answer from '<think>...</think> Answer: $num' format."""
    m = _FORMAT_RE.match(text)
    if m:
        return m.group(2).strip()
    # Fallback: try plain 'Answer: ...'
    match = re.search(r"Answer:\s*(.+)", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _check_format(text: str) -> bool:
    """Check if text matches <think>cot</think> Answer: $num format."""
    return _FORMAT_RE.match(text) is not None


def load_environment(n: int = 100000, **kwargs) -> vf.Environment:
    parser = vf.Parser(extract_fn=extract_answer)

    def reward_func(parser, completion, answer, **kwargs):
        """Reward 1.0 only if format is correct AND answer matches exactly."""
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        else:
            text = completion
        if not _check_format(text):
            return 0.0
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    rubric = vf.Rubric(
        parser=parser,
        funcs=[reward_func],
        weights=[1.0],
    )

    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    if n > 0 and n < len(ds):
        ds = ds.shuffle(seed=42).select(range(n))

    def preprocess(example):
        prompt = example["prompt"]
        # Replace format instructions to require <think> tags
        if isinstance(prompt, list):
            for msg in prompt:
                if msg.get("role") == "user":
                    msg["content"] = (
                        msg["content"].rstrip()
                        + "\n\nPut your reasoning inside <think>...</think> tags, "
                        "then write your final answer as: Answer: <your answer>"
                    )
        return {
            "prompt": prompt,
            "answer": example["reward_model"]["ground_truth"],
        }

    ds = ds.map(preprocess)

    env = vf.SingleTurnEnv(
        dataset=ds,
        parser=parser,
        rubric=rubric,
    )
    return env
