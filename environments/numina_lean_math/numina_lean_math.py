import re

from datasets import Dataset, load_dataset

import verifiers as vf

_FORMAT_RE = re.compile(r"^\s*<think>(.+?)</think>\s*Answer:\s*(.+?)\s*$", re.DOTALL)
_ALLOWED_TYPES = {"math-word-problem", "MCQ"}
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+|\d+\.\d+|\d+/\d+)$")


def extract_answer(text: str) -> str:
    match = _FORMAT_RE.match(text)
    if match:
        return match.group(2).strip()
    fallback = re.search(r"Answer:\s*(.+)", text.strip(), re.IGNORECASE)
    return fallback.group(1).strip() if fallback else ""


def _build_prompt(problem: str) -> list[dict[str, str]]:
    instruction = (
        "Solve the following math problem step by step. Put your reasoning inside "
        "<think>...</think> tags. On the final line, write exactly: "
        "Answer: <your answer>."
    )
    return [{"role": "user", "content": f"{instruction}\n\n{problem.strip()}"}]


def _is_simple_numeric_answer(answer: str) -> bool:
    normalized = answer.replace(",", "").replace(" ", "")
    return _NUMERIC_RE.fullmatch(normalized) is not None


def _prepare_dataset(seed: int) -> Dataset:
    ds = load_dataset("AI-MO/NuminaMath-LEAN", split="train")

    rows: list[dict[str, object]] = []
    for ex in ds:
        question_type = ex.get("question_type")
        problem = ex.get("problem")
        answer = ex.get("answer")

        if question_type not in _ALLOWED_TYPES:
            continue
        if not isinstance(problem, str) or problem.strip() == "":
            continue
        if answer is None:
            continue

        answer_str = str(answer).strip()
        if answer_str == "":
            continue
        if not _is_simple_numeric_answer(answer_str):
            continue

        rows.append(
            {
                "prompt": _build_prompt(problem),
                "answer": answer_str,
                "question_type": question_type,
                "source": ex.get("source"),
                "problem_idx": ex.get("uuid"),
            }
        )

    dataset = Dataset.from_list(rows)
    return dataset.shuffle(seed=seed)


def load_environment(eval_size: int = 1000, seed: int = 42) -> vf.Environment:
    parser = vf.Parser(extract_fn=extract_answer)

    def answer_reward_func(parser, completion, answer, **kwargs):
        text = (
            completion[-1].get("content", "")
            if isinstance(completion, list)
            else completion
        )
        if not isinstance(text, str):
            return 0.0
        if _FORMAT_RE.match(text) is None:
            return 0.0
        pred = parser.parse_answer(completion) or ""
        return 1.0 if pred.strip() == str(answer).strip() else 0.0

    rubric = vf.Rubric(parser=parser, funcs=[answer_reward_func], weights=[1.0])

    dataset = _prepare_dataset(seed=seed)

    if eval_size < 0:
        raise ValueError("eval_size must be >= 0")
    eval_size = min(eval_size, len(dataset))

    eval_dataset = (
        dataset.select(range(eval_size)) if eval_size > 0 else Dataset.from_list([])
    )
    train_dataset = dataset.select(range(eval_size, len(dataset)))

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
    )
