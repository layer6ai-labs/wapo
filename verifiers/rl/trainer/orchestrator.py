import asyncio
import logging
import queue
import threading
import time
from typing import Any

import httpx
import numpy as np
from datasets import Dataset
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase

from verifiers import Environment


class Microbatch(BaseModel):
    """Microbatch for batch generation"""

    input_ids: list[list[int]]
    loss_mask: list[list[int]]
    sampling_logprobs: list[list[float]]
    advantages: list[list[float]]
    seq_loss_weights: list[float]
    group_ids: list[int]
    items: int


class Batch(BaseModel):
    """Result from batch generation"""

    batch_id: int
    microbatches: list[list[Microbatch]]
    items_per_process: list[int]
    global_item_count: int
    num_groups_with_pos_adv: int = 0
    num_groups_with_neg_adv: int = 0
    loss_normalizer: float = 1.0
    # logging
    generation_time: float = 0.0
    prompts: list[Any] = Field(default_factory=list)
    completions: list[Any] = Field(default_factory=list)
    metrics_dict: dict[str, float] = Field(default_factory=dict)
    rewards_dict: dict[str, list[float]] = Field(default_factory=dict)


class Orchestrator:
    """
    Manages asynchronous batch generation in parallel with RL training.
    """

    def __init__(
        self,
        env: Environment,
        client_base_url: str,
        client_api_key: str,
        client_limit: int,
        client_timeout: float,
        model_name: str,
        sampling_args: dict[str, Any],
        rollouts_per_example: int,
        batch_size: int,
        micro_batch_size: int,
        num_processes: int,
        generation_timeout: float,
        processing_class: PreTrainedTokenizerBase,
        mask_env_responses: bool,
        max_seq_len: int,
        max_prompt_len: int,
        mask_truncated_completions: bool,
        zero_truncated_completions: bool,
        max_concurrent: int,
        loss_type: str = "gspo",
        loss_mean_level: str | None = None,
        mask_negative_advantages: bool = False,
        mask_positive_advantages: bool = False,
    ):
        self.env = env
        self.client_base_url = client_base_url
        self.client_api_key = client_api_key
        self.client_limit = client_limit
        self.client_timeout = client_timeout
        self.client = None  # created in worker thread
        self.model_name = model_name
        self.sampling_args = sampling_args
        self.rollouts_per_example = rollouts_per_example
        self.prompts_per_batch = batch_size // rollouts_per_example
        self.micro_batch_size = micro_batch_size
        self.num_processes = num_processes
        self.generation_timeout = generation_timeout
        self.processing_class = processing_class
        self.mask_env_responses = mask_env_responses
        self.max_seq_len = max_seq_len
        self.max_prompt_len = max_prompt_len
        self.mask_truncated_completions = mask_truncated_completions
        self.zero_truncated_completions = zero_truncated_completions
        self.max_concurrent = max_concurrent
        self.loss_type = loss_type
        self.loss_mean_level = loss_mean_level
        self.mask_negative_advantages = mask_negative_advantages
        self.mask_positive_advantages = mask_positive_advantages
        # Rewards are now a weighted average, so max possible reward is 1.0
        self.max_reward = 1.0

        # queues for communication
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.eval_result_queue: queue.Queue = queue.Queue()
        self.is_generating = False
        self.completed_batches = {}

        self.worker_thread = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)
        self.is_generating = False
        self.worker_loop = None

        max_length = self.max_prompt_len
        assert env.dataset is not None

        def filter_by_prompt_length(example, processing_class):
            prompt = example["prompt"]
            if isinstance(prompt, list):
                prompt_text = processing_class.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = prompt
            prompt_ids = processing_class.encode(prompt_text)
            return len(prompt_ids) <= max_length

        env.dataset = env.dataset.filter(
            filter_by_prompt_length,
            fn_kwargs={"processing_class": processing_class},
        )

    def get_dataset_slice(self, batch_id: int) -> Dataset:
        """Get dataset slice for a given batch id"""
        num_rows = self.prompts_per_batch
        dataset = self.env.get_dataset()
        total_rows = len(dataset)
        if total_rows == 0:
            raise ValueError("Environment dataset is empty")
        offset = (batch_id * num_rows) % total_rows
        indices = [(offset + i) % total_rows for i in range(num_rows)]
        return dataset.select(indices)

    def start(self):
        """Start the async generation worker thread"""
        self.worker_thread = threading.Thread(
            target=self.generation_worker, daemon=True, name="BatchGenerator"
        )
        self.worker_thread.start()

    def stop(self):
        """Stop the async generation worker thread"""
        self.stop_event.set()
        self.request_queue.put(None)  # poison pill
        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)

    def submit_batch(self, batch_id: int):
        self.request_queue.put(batch_id)

    def get_batch(self, batch_id: int) -> Batch:
        """
        Get a completed batch result. Blocks until the batch is ready.

        Args:
            batch_id: The batch ID to retrieve
            timeout: Maximum time to wait

        Returns:
            BatchResult: The completed batch result

        Raises:
            TimeoutError: batch doesn't complete within timeout
            RuntimeError: generation failed
        """
        timeout = self.generation_timeout
        start_time = time.time()
        while True:
            if batch_id in self.completed_batches:
                return self.completed_batches.pop(batch_id)
            try:
                result = self.result_queue.get(timeout=0.1)
                self.completed_batches[result.batch_id] = result
                if result.batch_id == batch_id:
                    return self.completed_batches.pop(batch_id)
            except queue.Empty:
                pass

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Batch {batch_id} timed out after {timeout}s")

    def submit_eval(self, num_examples: int = -1, sampling_args: dict | None = None):
        """Submit an eval request to the worker thread."""
        self.request_queue.put(
            (
                "eval",
                {
                    "num_examples": num_examples,
                    "sampling_args": sampling_args,
                },
            )
        )

    def get_eval_result(self, timeout: float | None = None):
        """Block until eval result is ready."""
        timeout = timeout or self.generation_timeout
        return self.eval_result_queue.get(timeout=timeout)

    async def run_eval(self, num_examples: int = -1, sampling_args: dict | None = None):
        """Run evaluation on the worker thread's event loop."""
        assert self.client is not None
        eval_inputs = self.env.get_eval_inputs(num_examples=num_examples)
        results = await self.env.a_generate(
            eval_inputs,
            client=self.client,
            model=self.model_name,
            sampling_args=sampling_args or self.sampling_args,
            score_rollouts=True,
            max_concurrent=self.max_concurrent,
        )
        return results

    def generation_worker(self):
        """Worker thread that processes generation requests"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.worker_loop = loop
        self.client = AsyncOpenAI(
            base_url=self.client_base_url,
            api_key=self.client_api_key,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=self.client_limit),
                timeout=self.client_timeout,
            ),
        )
        try:
            while not self.stop_event.is_set():
                try:
                    request = self.request_queue.get(timeout=0.1)
                    if request is None:  # poison pill
                        break
                    if isinstance(request, tuple) and request[0] == "eval":
                        eval_kwargs = request[1]
                        eval_result = loop.run_until_complete(
                            self.run_eval(**eval_kwargs)
                        )
                        self.eval_result_queue.put(eval_result)
                    else:
                        batch_id = request
                        result = loop.run_until_complete(self.generate_batch(batch_id))
                        self.result_queue.put(result)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in generation worker: {e}")
                    raise e
        finally:
            loop.run_until_complete(self.client.close())
            loop.close()
            asyncio.set_event_loop(None)

    async def generate_batch(self, batch_id: int) -> Batch:
        """
        Generate a single batch asynchronously.
        """
        # MODIFIED: log when batch generation starts and ends
        self.logger.info(f"Start generating batch {batch_id}")
        self.is_generating = True
        assert self.client is not None
        start_time = time.time()
        batch_ds = self.get_dataset_slice(batch_id)
        repeated_ds = batch_ds.repeat(self.rollouts_per_example)
        env_results = await self.env.a_generate(
            repeated_ds,
            client=self.client,
            model=self.model_name,
            sampling_args=self.sampling_args,
            score_rollouts=True,
            max_concurrent=self.max_concurrent,
        )
        self.is_generating = False
        self.logger.info(f"Finish generating batch {batch_id}")
        wall_clock_s = time.time() - start_time

        processed_results = self.env.process_env_results_vllm(
            prompts=env_results.prompt,
            completions=env_results.completion,
            states=env_results.state,
            rewards=env_results.reward,
            processing_class=self.processing_class,
            max_seq_len=self.max_seq_len,
            mask_env_responses=self.mask_env_responses,
            mask_truncated_completions=self.mask_truncated_completions,
            zero_truncated_completions=self.zero_truncated_completions,
        )

        rewards_dict = {"reward": processed_results.rewards}
        for k in env_results.metrics:
            rewards_dict[k] = env_results.metrics[k]

        rewards: list[float] = processed_results.rewards
        advantages: list[float] = [0.0] * len(rewards)
        prompts_in_batch = len(batch_ds)
        for prompt_idx in range(prompts_in_batch):
            group_indices = [
                prompt_idx + k * prompts_in_batch
                for k in range(self.rollouts_per_example)
                if (prompt_idx + k * prompts_in_batch) < len(rewards)
            ]
            if not group_indices:
                continue
            group = [rewards[i] for i in group_indices]
            gmean = sum(group) / float(len(group))

            # Advantage normalization determined by loss type
            if self.loss_type in ("grpo", "dapo", "gspo"):
                # std-normalized: (r - mean) / (std + eps)
                gstd = (sum((r - gmean) ** 2 for r in group) / len(group)) ** 0.5
                for idx, r in zip(group_indices, group):
                    advantages[idx] = (r - gmean) / (gstd + 1e-8)
            elif self.loss_type in ("raft", "raft++"):
                # raw reward as advantage
                for idx, r in zip(group_indices, group):
                    advantages[idx] = r
            else:  # dr_grpo, dr_dapo, dr_dapo_seq, wapo
                # mean-centered, no normalization
                for idx, r in zip(group_indices, group):
                    advantages[idx] = r - gmean

        # Compute per-prompt overlap metrics across all rollout pairs
        # Split by pair type: pos-neg, pos-pos, neg-neg
        overlap_stats: dict[str, list[float]] = {
            "divergence_pos_neg": [],
            "divergence_pos_pos": [],
            "divergence_neg_neg": [],
            "jaccard_pos_neg": [],
            "jaccard_pos_pos": [],
            "jaccard_neg_neg": [],
            "contested_pos_neg": [],
            "contested_pos_pos": [],
            "contested_neg_neg": [],
        }
        for prompt_idx in range(prompts_in_batch):
            group_indices = [
                prompt_idx + k * prompts_in_batch
                for k in range(self.rollouts_per_example)
                if (prompt_idx + k * prompts_in_batch) < len(rewards)
            ]
            if len(group_indices) < 2:
                continue
            # Precompute full token sequences for each rollout in group
            group_ids = [
                processed_results.prompt_ids[i] + processed_results.completion_ids[i]
                for i in group_indices
            ]
            group_advs = [advantages[i] for i in group_indices]
            for a in range(len(group_indices)):
                for b in range(a + 1, len(group_indices)):
                    ids_a, ids_b = group_ids[a], group_ids[b]
                    adv_a, adv_b = group_advs[a], group_advs[b]
                    # Determine pair type
                    if adv_a > 0 and adv_b > 0:
                        suffix = "pos_pos"
                    elif adv_a < 0 and adv_b < 0:
                        suffix = "neg_neg"
                    elif (adv_a > 0 and adv_b < 0) or (adv_a < 0 and adv_b > 0):
                        suffix = "pos_neg"
                    else:
                        continue  # skip zero-advantage pairs
                    # Divergence point
                    min_len = min(len(ids_a), len(ids_b))
                    max_len = max(len(ids_a), len(ids_b))
                    divergence_pos = min_len
                    for pos in range(min_len):
                        if ids_a[pos] != ids_b[pos]:
                            divergence_pos = pos
                            break
                    if max_len > 0:
                        overlap_stats[f"divergence_{suffix}"].append(
                            divergence_pos / max_len
                        )
                    # Jaccard similarity
                    set_a, set_b = set(ids_a), set(ids_b)
                    union = set_a | set_b
                    if union:
                        overlap_stats[f"jaccard_{suffix}"].append(
                            len(set_a & set_b) / len(union)
                        )
                    # Contested (position-wise agreement)
                    if min_len > 0:
                        contested = sum(
                            1 for pos in range(min_len) if ids_a[pos] == ids_b[pos]
                        )
                        overlap_stats[f"contested_{suffix}"].append(contested / min_len)

        # Track groups with at least one correct answer and masking fractions
        groups_with_correct = 0
        groups_all_neg_masked = 0  # all rollouts have adv <= 0 (no pos signal)
        groups_all_pos_masked = 0  # all rollouts have adv >= 0 (no neg signal)
        total_neg_adv_seqs = 0
        for prompt_idx in range(prompts_in_batch):
            group_indices = [
                prompt_idx + k * prompts_in_batch
                for k in range(self.rollouts_per_example)
                if (prompt_idx + k * prompts_in_batch) < len(rewards)
            ]
            if not group_indices:
                continue
            if any(rewards[i] > 0 for i in group_indices):
                groups_with_correct += 1
            if all(advantages[i] <= 0 for i in group_indices):
                groups_all_neg_masked += 1
            if all(advantages[i] >= 0 for i in group_indices):
                groups_all_pos_masked += 1
            total_neg_adv_seqs += sum(1 for i in group_indices if advantages[i] <= 0)

        metrics_dict = {}
        if prompts_in_batch > 0:
            metrics_dict["groups/with_correct"] = float(
                groups_with_correct / prompts_in_batch
            )
            metrics_dict["groups/all_neg_masked"] = float(
                groups_all_neg_masked / prompts_in_batch
            )
            metrics_dict["groups/all_pos_masked"] = float(
                groups_all_pos_masked / prompts_in_batch
            )
        if len(rewards) > 0:
            metrics_dict["masking/neg_adv_seq_fraction"] = float(
                total_neg_adv_seqs / len(rewards)
            )
        for key, values in overlap_stats.items():
            if values:
                metrics_dict[f"overlap/{key}"] = float(np.mean(values))
        if rewards:
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            metrics_dict["reward"] = float(rewards_arr.mean())
            metrics_dict["reward/std"] = float(rewards_arr.std())

        if advantages:
            adv_arr = np.asarray(advantages, dtype=np.float32)
            metrics_dict["advantage/absmean"] = float(np.abs(adv_arr).mean())

        for reward_name, values in env_results.metrics.items():
            if len(values) == 0:
                continue
            reward_values = np.asarray(values, dtype=np.float32)
            metrics_dict[f"reward/{reward_name}"] = float(reward_values.mean())

        completion_lengths = [len(ids) for ids in processed_results.completion_ids]
        if completion_lengths:
            completion_lengths_arr = np.asarray(completion_lengths, dtype=np.float32)
            metrics_dict["tokens/completion"] = float(completion_lengths_arr.mean())

            completion_mask_lengths = np.asarray(
                [sum(mask) for mask in processed_results.completion_mask],
                dtype=np.float32,
            )
            valid_tokens = completion_mask_lengths.sum()
            total_tokens = completion_lengths_arr.sum()
            if total_tokens > 0:
                masked_fraction = 1.0 - (valid_tokens / total_tokens)
                metrics_dict["tokens/masked_fraction"] = float(masked_fraction)

        generation_ms: list[float] = []
        scoring_ms: list[float] = []
        total_ms: list[float] = []
        for state in env_results.state:
            timing = state.get("timing", {})
            if "generation_ms" in timing:
                generation_ms.append(float(timing["generation_ms"]))
            if "scoring_ms" in timing:
                scoring_ms.append(float(timing["scoring_ms"]))
            if "total_ms" in timing:
                total_ms.append(float(timing["total_ms"]))

        if generation_ms:
            metrics_dict["timing/generation_ms"] = float(np.mean(generation_ms))
        if scoring_ms:
            metrics_dict["timing/scoring_ms"] = float(np.mean(scoring_ms))
        if total_ms:
            metrics_dict["timing/total_ms"] = float(np.mean(total_ms))

        metrics_dict["wall_clock/generate_s"] = float(wall_clock_s)

        # build per-process microbatches
        N = len(processed_results.rewards)

        # Compute per-token weights and outer normalizer for loss averaging
        seq_loss_weights = [1.0] * N
        seq_token_counts = [
            sum(processed_results.prompt_mask[i])
            + sum(processed_results.completion_mask[i])
            for i in range(N)
        ]

        # Determine effective mean level from config override or loss_type default
        _DEFAULT_MEAN_LEVELS = {
            "grpo": "seq",
            "gspo": "seq",
            "dr_dapo_seq": "seq",
            "raft": "seq",
            "raft++": "seq",
            "dapo": "group",
            "dr_dapo": "group",
            "wapo": "group",
            "dr_grpo": "global",
        }
        effective_mean_level = self.loss_mean_level or _DEFAULT_MEAN_LEVELS.get(
            self.loss_type, "seq"
        )

        def _is_contributing(idx):
            if self.mask_negative_advantages and advantages[idx] <= 0:
                return False
            if self.mask_positive_advantages and advantages[idx] >= 0:
                return False
            return True

        if effective_mean_level == "seq":
            for i in range(N):
                seq_loss_weights[i] = 1.0 / max(seq_token_counts[i], 1)
            contributing_seqs = sum(1 for i in range(N) if _is_contributing(i))
            loss_normalizer = float(max(contributing_seqs, 1))

        elif effective_mean_level == "group":
            contributing_groups = 0
            for prompt_idx in range(prompts_in_batch):
                group_indices = [
                    prompt_idx + k * prompts_in_batch
                    for k in range(self.rollouts_per_example)
                    if (prompt_idx + k * prompts_in_batch) < N
                ]
                num_pos = sum(1 for i in group_indices if _is_contributing(i))
                if self.loss_type == "wapo":
                    # Denominator: group_size * max_seq_len
                    denom = len(group_indices) * self.max_seq_len
                    for idx in group_indices:
                        seq_loss_weights[idx] = 1.0 / denom
                    if num_pos > 0:
                        contributing_groups += 1
                else:
                    contributing_tokens = sum(
                        seq_token_counts[i]
                        for i in group_indices
                        if _is_contributing(i)
                    )
                    for idx in group_indices:
                        seq_loss_weights[idx] = 1.0 / max(contributing_tokens, 1)
                    if contributing_tokens > 0:
                        contributing_groups += 1
            loss_normalizer = float(max(contributing_groups, 1))

        else:  # "global"
            # seq_loss_weights stay at 1.0
            if self.loss_type == "dr_grpo" and self.loss_mean_level is None:
                # Dr.GRPO paper: normalize by N * max_gen_len
                max_gen_len = max(seq_token_counts) if seq_token_counts else 1
                loss_normalizer = float(N * max_gen_len)
            else:
                contributing_tokens = sum(
                    seq_token_counts[i] for i in range(N) if _is_contributing(i)
                )
                loss_normalizer = float(max(contributing_tokens, 1))

        per_proc = N // self.num_processes
        microbatches: list[list[Microbatch]] = []
        items_per_process: list[int] = []
        for proc in range(self.num_processes):
            ps = proc * per_proc
            pe = ps + per_proc
            proc_mbs: list[Microbatch] = []
            proc_item_total = 0
            for s in range(ps, pe, self.micro_batch_size):
                e = min(s + self.micro_batch_size, pe)
                ids_chunk = [
                    processed_results.prompt_ids[i]
                    + processed_results.completion_ids[i]
                    for i in range(s, e)
                ]
                mask_chunk = [
                    processed_results.prompt_mask[i]
                    + processed_results.completion_mask[i]
                    for i in range(s, e)
                ]
                slogp_chunk = [
                    [0.0] * len(processed_results.prompt_mask[i])
                    + processed_results.completion_logprobs[i]
                    for i in range(s, e)
                ]
                lengths = [len(mask) for mask in mask_chunk]
                adv_chunk = [
                    [advantages[i]] * lengths[idx]
                    for idx, i in enumerate(list(range(s, e)))
                ]
                weight_chunk = [seq_loss_weights[i] for i in range(s, e)]
                group_chunk = [i % prompts_in_batch for i in range(s, e)]
                mb_items = sum(sum(mask) for mask in mask_chunk)
                microbatch = Microbatch(
                    input_ids=ids_chunk,
                    loss_mask=mask_chunk,
                    sampling_logprobs=slogp_chunk,
                    advantages=adv_chunk,
                    seq_loss_weights=weight_chunk,
                    group_ids=group_chunk,
                    items=mb_items,
                )
                proc_item_total += mb_items
                proc_mbs.append(microbatch)
            microbatches.append(proc_mbs)
            items_per_process.append(proc_item_total)

        global_item_count = sum(items_per_process)

        return Batch(
            batch_id=batch_id,
            microbatches=microbatches,
            items_per_process=items_per_process,
            global_item_count=global_item_count,
            num_groups_with_pos_adv=prompts_in_batch - groups_all_neg_masked,
            num_groups_with_neg_adv=prompts_in_batch - groups_all_pos_masked,
            loss_normalizer=loss_normalizer,
            generation_time=wall_clock_s,
            rewards_dict=rewards_dict,
            completions=env_results.completion,
            prompts=env_results.prompt,
            metrics_dict=metrics_dict,
        )
