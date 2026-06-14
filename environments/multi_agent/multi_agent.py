"""
Multi-Agent Environment for RL Training.

Usage:
    vf-rl @ configs/multi_agent.toml

Config example (configs/multi_agent.toml):
    model = "Qwen/Qwen2.5-1.5B-Instruct"

    [env]
    id = "multi_agent"

    [env.args]
    retriever_url = "http://localhost:8000/retrieve"
    answer_agent_url = "http://localhost:8001/v1"
    answer_agent_model = "Qwen/Qwen3-8B"
    train_dataset_path = "/path/to/train.json"
    eval_dataset_path = "/path/to/eval.json"

    [trainer.args]
    batch_size = 16
    max_steps = 100
"""

import collections
import re
import string
from typing import Optional

from datasets import load_dataset

import verifiers as vf
from verifiers.envs.multi_agent_env import MultiAgentEnv
from environments.multi_agent.agents.multi_agent_config import (
    AnswerAgentConfig,
    MultiAgentConfig,
    RetrieverConfig,
)


# ============================================================================
# Reward Helper Functions (same as vf_rag_agent)
# ============================================================================


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> list:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


# ============================================================================
# Main Environment Loader
# ============================================================================


def load_environment(
    # Retriever configuration
    retriever_url: str = "http://localhost:8000/retrieve",
    top_k: int = 3,
    # Answer agent configuration
    answer_agent_url: str = "http://localhost:8001/v1",
    answer_agent_model: str = "Qwen/Qwen3-8B",
    answer_agent_api_key: str = "EMPTY",
    # Dataset configuration - JSON file paths (like vf_rag_agent)
    train_dataset_path: Optional[str] = None,
    eval_dataset_path: Optional[str] = None,
    # Fallback: HuggingFace dataset (if paths not provided)
    dataset_name: str = "hotpotqa",
    dataset_split: str = "train",
    dataset_config: str = "distractor",
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    # Multi-agent configuration
    max_turns: int = 10,
    # System prompts (optional overrides)
    planning_system_prompt: str = "",
    answer_system_prompt: str = "",
):
    """
    Load the multi-agent environment.

    Args:
        retriever_url: URL of the retrieval server endpoint.
        top_k: Number of documents to retrieve.
        answer_agent_url: URL of the answer agent LLM server.
        answer_agent_model: Model name for the answer agent.
        answer_agent_api_key: API key for the answer agent.
        train_dataset_path: Path to training dataset JSON file.
        eval_dataset_path: Path to evaluation dataset JSON file.
        dataset_name: HuggingFace dataset name (fallback if paths not provided).
        dataset_split: Dataset split to use for training.
        dataset_config: Dataset configuration/subset name.
        num_train_examples: Number of training examples (-1 for all).
        num_eval_examples: Number of eval examples (-1 for all).
        max_turns: Maximum turns for the planning agent.
        planning_system_prompt: Custom system prompt for planning agent.
        answer_system_prompt: Custom system prompt for answer agent.

    Returns:
        MultiAgentEnv instance.
    """
    # ========================================================================
    # Load Dataset (JSON files preferred, fallback to HuggingFace)
    # ========================================================================
    if train_dataset_path:
        # Load from JSON file (like vf_rag_agent)
        dataset = load_dataset("json", data_files=train_dataset_path, split="train")
    else:
        # Fallback to HuggingFace
        try:
            dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
        except Exception:
            dataset = load_dataset(dataset_name, split=dataset_split)

    if num_train_examples > 0:
        dataset = dataset.select(range(min(num_train_examples, len(dataset))))

    # Note: Do NOT rename "question" to "prompt" here.
    # Let Environment.prepare_dataset() handle it - it will format
    # the question as messages with the system prompt.

    # Load eval dataset
    eval_dataset = None
    if eval_dataset_path:
        # Load from JSON file (like vf_rag_agent)
        eval_dataset = load_dataset("json", data_files=eval_dataset_path, split="train")
    else:
        # Fallback to HuggingFace
        try:
            eval_split = "validation" if dataset_split == "train" else "test"
            try:
                eval_dataset = load_dataset(
                    dataset_name, dataset_config, split=eval_split
                )
            except Exception:
                eval_dataset = load_dataset(dataset_name, split=eval_split)
        except Exception:
            pass  # No eval dataset available

    if eval_dataset is not None:
        if num_eval_examples > 0:
            eval_dataset = eval_dataset.select(
                range(min(num_eval_examples, len(eval_dataset)))
            )
        # Note: Do NOT rename columns - let prepare_dataset() handle it

    # ========================================================================
    # Create Agent Configs
    # ========================================================================
    retriever_config = RetrieverConfig(
        url=retriever_url,
        top_k=top_k,
    )

    answer_config_kwargs = {
        "base_url": answer_agent_url,
        "model_name": answer_agent_model,
        "api_key": answer_agent_api_key,
        "retriever": retriever_config,
    }
    if answer_system_prompt:
        answer_config_kwargs["system_prompt"] = answer_system_prompt

    answer_agent_config = AnswerAgentConfig(**answer_config_kwargs)

    # Configure multi-agent
    multi_config_kwargs = {
        "answer_agent": answer_agent_config,
        "max_turns": max_turns,
    }
    if planning_system_prompt:
        multi_config_kwargs["planning_system_prompt"] = planning_system_prompt

    multi_agent_config = MultiAgentConfig(**multi_config_kwargs)

    # ========================================================================
    # Reward Functions
    # ========================================================================

    def _completion_to_text(completion) -> str:
        """Convert completion (list of messages or string) to text."""
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list):
            # Extract content from all assistant messages
            texts = []
            for msg in completion:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    texts.append(msg.get("content", ""))
            return "\n".join(texts)
        return ""

    def _extract_pred_answer(completion) -> Optional[str]:
        """Extract predicted answer from <answer> tags."""
        text = _completion_to_text(completion)
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def f1_reward(completion, answer, state, **kwargs) -> float:
        """F1 score reward (same as vf_rag_agent)."""
        pred_answer = _extract_pred_answer(completion)
        if pred_answer is None:
            # Try from state["final_answer"] (set by MultiAgentEnv.get_final_answer)
            pred_answer = state.get("final_answer")
        if pred_answer is not None:
            return compute_f1(answer, pred_answer)
        return 0.0

    # Create rubric with F1 reward only
    rubric = vf.Rubric(
        funcs=[f1_reward],
        weights=[1.0],
    )

    # ========================================================================
    # Create Environment
    # ========================================================================
    env = MultiAgentEnv(
        multi_agent_config=multi_agent_config,
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
    )

    return env
