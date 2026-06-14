import verifiers as vf
from verifiers.types import ChatMessage, Messages, State
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.rubric import Rubric
from datasets import load_dataset
import requests
import re
import string
import collections


class RAGAgentEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        dataset,
        eval_dataset=None,
        system_prompt: str | None = None,
        parser: XMLParser = XMLParser(
            fields=["think", ("search", "answer")], answer_field="answer"
        ),
        rubric: Rubric = Rubric(),
        max_turns: int = 5,
        retrieve_url: str = "http://192.168.4.147:8080/retrieve",
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            message_type="chat",
            max_turns=max_turns,
            **kwargs,
        )
        self.parser = parser
        self.rubric = rubric
        self.retrieve_url = retrieve_url

    def setup_state(self, state: State, **kwargs) -> State:
        if state.get("turn", 0) == 0:
            state["wrong_format"] = False
            state["done"] = False
            state["retrieved_tables"] = set()
            state["retrieved_passages"] = set()
        return state

    async def env_response(self, messages: Messages, state: State, **kwargs):
        last_msg = messages[-1]
        if last_msg["role"] == "assistant":
            llm_output = last_msg["content"]
        else:
            return [], state  # No response if not assistant message

        if await self.max_turns_reached(state):
            state["done"] = True
            return [], state

        parsed_output = self.parser.parse(llm_output)
        if parsed_output.answer:
            state["done"] = True
            state["pred_answer"] = parsed_output.answer
            return [], state
        elif parsed_output.search:
            query = parsed_output.search

            # Do the search, maybe using api
            data = {"queries": [query], "top_k": 3}

            retrieved = requests.post(self.retrieve_url, json=data).json()[0]
            retrieved = [re.sub(r"\s+", " ", x) for x in retrieved]
            retrieved = [re.sub(r"(\n\s*)+", "\n", x) for x in retrieved]
            for context in retrieved:
                if "Markdown Table:" in context:
                    state["retrieved_tables"].add(context.split(" ")[0])
                else:
                    state["retrieved_passages"].add("wiki/" + context.split(" ")[0])

            retrieved_content = "\n\n".join(retrieved)
            response = [
                {
                    "role": "user",
                    "content": f"<information>{retrieved_content}</information>",
                }
            ]

            return response, state
        else:
            state["wrong_format"] = True
            state["done"] = True
            return [], state

    def is_completed(self, messages, state, **kwargs):
        return state.get("done", False)


def load_environment(
    train_dataset_path: str = "/home/zhichen/ott-qa/mix-10000-v1.json",
    eval_dataset_path: str = "/home/zhichen/ott-qa/ott-eval.json",
    retrieve_url: str = "http://192.168.4.147:8080/retrieve",
    reward_weights: list[float] | None = None,
    **kwargs,
) -> vf.Environment:
    """
    Loads a custom environment.
    """

    MAX_HOPS = 10

    def normalize_answer(s):
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

    def retrieve_reward(answer, state):
        f1_score = f1_reward(answer, state)
        gt_passages = set(state["info"]["gt_passages"])
        passage_recall = len(
            gt_passages.intersection(state["retrieved_passages"])
        ) / len(gt_passages)
        table_recall = (
            1 if state["info"]["gt_table"] in state["retrieved_tables"] else 0
        )
        retrieve_reward = (passage_recall + table_recall) / 2
        if f1_score <= 0.5:
            return retrieve_reward * 0.2
        else:
            return retrieve_reward * f1_score

    def exact_match_reward(prompt, completion, answer, state):
        """Reward based on exact match with the answer."""

        def compute_exact(a_gold, a_pred):
            return int(normalize_answer(a_gold) == normalize_answer(a_pred))

        if "pred_answer" in state and compute_exact(answer, state["pred_answer"]) == 1:
            return 1.0
        else:
            return 0.0

    def get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def f1_reward(answer, state):
        if "pred_answer" in state:
            return compute_f1(answer, state["pred_answer"])
        return 0

    def em_reward(answer, state):
        if "pred_answer" in state:
            return float(
                normalize_answer(answer) == normalize_answer(state["pred_answer"])
            )
        return 0

    def turn_reward(answer, state, completion):
        """Reward based on number of turns taken to answer the question."""
        # turns_taken = state.get("turn", 0)
        model_messages = [m for m in completion if m.get("role") == "assistant"]
        turns_taken = len(model_messages)
        if "pred_answer" in state:
            return turns_taken / MAX_HOPS
        return 0

    # def format_reward_func_v1(completion: list[ChatMessage]) -> float:
    #     model_messages = [m for m in completion if m.get("role") == "assistant"]
    #     if not model_messages:
    #         return 0.0
    #     def _count(tag, text):
    #         return text.count(f"<{tag}>")
    #     def _count_close(tag, text):
    #         return text.count(f"</{tag}>")
    #     scores = []
    #     for msg in model_messages:
    #         content = str(msg.get("content", ""))
    #         think_score = 0.5 if _count("think", content) == 1 and _count_close("think", content) == 1 else 0.0
    #         has_exact_search = _count("search", content) == 1 and _count_close("search", content) == 1
    #         has_exact_answer = _count("answer", content) == 1 and _count_close("answer", content) == 1
    #         end_score = 0.5 if (has_exact_search ^ has_exact_answer) else 0.0
    #         msg_score = think_score + end_score
    #         scores.append(msg_score)
    #     avg_score = sum(scores) / len(scores)
    #     if len(model_messages) == 1:
    #         avg_score -= 0.5
    #     return max(avg_score, 0.0)

    _FMT_SEARCH = re.compile(
        r"^\s*<think>(.*?)</think>\s*<search>(.*?)</search>\s*$", re.DOTALL
    )
    _FMT_ANSWER = re.compile(
        r"^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$", re.DOTALL
    )
    _HAS_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*>")

    def format_reward_func(completion: list[ChatMessage]) -> float:
        model_messages = [m for m in completion if m.get("role") == "assistant"]
        if not model_messages:
            return 0.0

        scores: list[float] = []
        for i, msg in enumerate(model_messages):
            content = str(msg.get("content", ""))
            is_last = i == len(model_messages) - 1

            # Intermediate turns: only <think>...<search>... allowed
            # Last turn: <think>...<search>... or <think>...<answer>... allowed
            m = _FMT_SEARCH.match(content)
            if not m and is_last:
                m = _FMT_ANSWER.match(content)

            if m and m.group(1).strip() and m.group(2).strip():
                # Reject if any XML-like tags appear inside the captured content
                if _HAS_TAG.search(m.group(1)) or _HAS_TAG.search(m.group(2)):
                    scores.append(0.0)
                else:
                    scores.append(1.0)
            else:
                scores.append(0.0)

        avg_score = sum(scores) / len(scores)
        if len(model_messages) == 1:
            avg_score = 0.0
        return max(avg_score, 0.0)

    parser = XMLParser(fields=["think", ("search", "answer")], answer_field="answer")
    format_reward_fn = format_reward_func
    reward_fns = [format_reward_fn, turn_reward, f1_reward, em_reward]
    default_weights = [0.3, 0.0, 1.0, 0.0]
    rubric = vf.Rubric(reward_fns, weights=reward_weights or default_weights)
    train_dataset = load_dataset("json", data_files=train_dataset_path, split="train")
    # train_dataset = train_dataset.select(range(100))
    eval_dataset = load_dataset("json", data_files=eval_dataset_path, split="train")
    eval_dataset = eval_dataset.select(range(100))
    MULTI_HOP_SYSTEM_PROMPT = (
        "You are a helpful assistant."
        "**Your task:**\n"
        "You need to answer complex questions by retrieving relevant information and reasoning step-by-step.\n"
        "You **must** conduct reasoning inside <think> and </think> in each of your response before you make a search or give the final answer. The reasoning should include analyzing the question, making a plan, and answering sub-questions by reasoning on existing content.\n"
        "You should obtain knowledge by calling a search engine by <search> query </search> when you come up with a suitable query. The responses from the search engine will be the top search results and will be between the tags <information> and </information>.\n"
        "Always ground your answer to the information the search engine returns. Do not use any information that is not returned by the search engine. You can call the search engine any number of times you want.\n"
        "Once enough information is gathered, produce a precise and concise answer to the original question and wrap it in <answer>...</answer>.\n\n"
    )

    return RAGAgentEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=MULTI_HOP_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        max_turns=MAX_HOPS,
        retrieve_url=retrieve_url,
        **kwargs,
    )
