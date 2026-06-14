import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import deepspeed
import httpx
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate.utils import (
    broadcast_object_list,
    is_peft_model,
)
from accelerate.utils.memory import clear_device_cache
from peft import PeftConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback

from openai import AsyncOpenAI

import verifiers as vf
import wandb
from verifiers.rl.inference.client import VLLMClient
from verifiers.rl.trainer.config import RLConfig
from verifiers.rl.trainer.orchestrator import Orchestrator
from verifiers.rl.trainer.utils import (
    entropy_from_logits,
    finalize_stat_tracker,
    init_stat_tracker,
    pad,
    prepare_peft_model,
    selective_log_softmax,
    summarize_values,
    update_stat_tracker,
)
from verifiers.types import Messages
from verifiers.utils.logging_utils import print_prompt_completions_sample
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls


# MODIFIED: Function to calculate LoRA parameters L2 norm squared
def calculate_lora_norm_squared(model):
    """
    Calculates the sum of the squared L2 norm of all trainable parameters.
    In a PEFT/LoRA model, these are the adapter weights (A and B matrices).
    """
    total_norm_sq = 0.0

    # Iterate over all named parameters in the model
    for name, param in model.named_parameters():
        # Check if the parameter is trainable (only LoRA layers should be True)
        # and has been initialized (has data)
        if param.requires_grad and param.data is not None:
            # .norm(2) calculates the L2 norm of the tensor
            # .item() extracts the value from the 0-dim tensor
            norm_sq = param.data.norm(2).pow(2).item()
            total_norm_sq += norm_sq

    return total_norm_sq


class OnPolicySyncCallback(TrainerCallback):
    """Syncs weights to vLLM and submits the next generation batch after
    optimizer.step(), making training fully on-policy.

    The HF Trainer loop is:
        training_step()  ->  optimizer.step()  ->  global_step += 1  ->  on_step_end()

    By doing update_vllm + submit_batch in on_step_end, the next batch is
    generated with post-optimizer weights, so the data is on-policy when
    consumed by the following training_step.
    """

    def __init__(self, rl_trainer: "RLTrainer"):
        self.rl_trainer = rl_trainer

    def on_step_end(self, args, state, control, **kwargs):
        self.rl_trainer.update_vllm()
        if self.rl_trainer.orchestrator:
            self.rl_trainer.orchestrator.submit_batch(state.global_step)


class RLTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel | str,
        env: vf.Environment,
        args: RLConfig,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)

        # model + tokenizer
        if isinstance(model, str):
            model_name = model
            model, processing_class = vf.get_model_and_tokenizer(
                model, use_liger=args.use_liger
            )
        else:
            model_name = model.config._name_or_path
        assert isinstance(model, PreTrainedModel)
        if args.use_lora and isinstance(args.lora_config, PeftConfig):
            model = prepare_peft_model(model, args.lora_config, args)
        model.warnings_issued["estimate_tokens"] = True  # suppress warning

        # Provide a dummy eval_dataset to satisfy HF Trainer's validation
        # when eval_strategy != "no". Actual eval is handled by our evaluate() override.
        if args.eval_strategy != "no" and "eval_dataset" not in kwargs:
            from datasets import Dataset as HFDataset

            kwargs["eval_dataset"] = HFDataset.from_dict({"labels": [0]})

        os.environ.setdefault("WANDB_CONSOLE", "off")
        super().__init__(
            model=model,
            args=args,
            processing_class=processing_class,
            **kwargs,
        )
        assert isinstance(self.processing_class, PreTrainedTokenizerBase)
        if self.processing_class.pad_token is None:
            self.processing_class.pad_token = self.processing_class.eos_token
        if self.processing_class.pad_token_id is None:
            self.processing_class.pad_token_id = self.processing_class.eos_token_id
        assert self.processing_class.pad_token_id is not None

        # env + model info (for evaluation)
        self.env = env
        self.model_name = model_name

        # batch args
        self.batch_size = args.batch_size
        self.max_steps = args.max_steps
        self.max_seq_len = args.max_seq_len
        self.temperature = args.temperature
        self.train_temperature = (
            args.train_temperature
            if args.train_temperature is not None
            else args.temperature
        )

        # loss args
        self.loss_type = args.loss_type
        self.clip_eps = args.clip_eps
        self.mask_ratio_low = args.mask_ratio_low
        self.mask_ratio_high = args.mask_ratio_high
        self.use_importance_sampling = args.use_importance_sampling
        self.token_bin_counting = args.token_bin_counting
        self.normalize_by_kept_tokens = args.normalize_by_kept_tokens
        self.include_bins = args.include_bins
        self.exclude_bins = args.exclude_bins
        self.loss_mean_level = args.loss_mean_level
        # Validate clip_eps for loss types that require it (not needed when IR is off)
        if (
            self.use_importance_sampling
            and self.loss_type
            in (
                "grpo",
                "dr_grpo",
                "gspo",
                "wapo",
            )
            and self.clip_eps is None
        ):
            raise ValueError(f"clip_eps is required for loss_type='{self.loss_type}'")
        if self.loss_type == "raft++" and self.clip_eps is None:
            raise ValueError("clip_eps is required for loss_type='raft++'")
        # Auto-configure masking
        self.mask_negative_advantages = args.mask_negative_advantages
        if self.loss_type in (
            "raft",
            "raft++",
            "wapo",
        ):
            self.mask_negative_advantages = True
        self.mask_positive_advantages = args.mask_positive_advantages
        self.num_mini_batches = (
            args.batch_size // args.mini_batch_size if args.mini_batch_size else 1
        )
        self.on_policy_sync = args.on_policy_sync

        # orchestrator (main process only)
        if self.accelerator.is_main_process:
            host = args.vllm_server_host
            port = args.vllm_server_port
            group_port = port + 43216  # derive NCCL group port from server port
            self.client = VLLMClient(
                host=host,
                port=port,
                group_port=group_port,
                connection_timeout=args.vllm_server_timeout,
            )
            self.client.init_communicator()
            vllm_base_url = f"http://{host}:{port}/v1"
            self.vllm_base_url = vllm_base_url
            self.orchestrator = Orchestrator(
                env=env,
                client_base_url=vllm_base_url,
                client_api_key="EMPTY",
                client_limit=args.max_concurrent,
                client_timeout=args.generation_timeout,
                model_name=model_name,
                sampling_args=dict(args.sampling_args),
                rollouts_per_example=args.rollouts_per_example,
                batch_size=args.batch_size,
                micro_batch_size=args.micro_batch_size,
                num_processes=self.accelerator.num_processes,
                generation_timeout=args.generation_timeout,
                processing_class=self.processing_class,
                mask_env_responses=args.mask_env_responses,
                max_seq_len=self.max_seq_len,
                max_prompt_len=args.max_prompt_len or self.max_seq_len,
                mask_truncated_completions=args.mask_truncated_completions,
                zero_truncated_completions=args.zero_truncated_completions,
                max_concurrent=args.max_concurrent,
                loss_type=args.loss_type,
                loss_mean_level=args.loss_mean_level,
                mask_negative_advantages=self.mask_negative_advantages,
                mask_positive_advantages=self.mask_positive_advantages,
            )
            self.orchestrator.start()

            # Determine initial batch_id: on resume, peek at trainer_state.json
            # to get global_step so the orchestrator generates the right batch.
            initial_batch_id = 0
            self._resume_checkpoint_path = None
            if args.resume_from_checkpoint:
                import json

                checkpoint_path = args.resume_from_checkpoint
                if checkpoint_path == "auto":
                    from transformers.trainer_utils import get_last_checkpoint

                    checkpoint_path = get_last_checkpoint(args.output_dir)
                    if checkpoint_path is None:
                        self.logger.warning(
                            "resume_from_checkpoint='auto' but no checkpoint found "
                            f"in {args.output_dir}. Starting from scratch."
                        )
                if checkpoint_path:
                    state_file = os.path.join(checkpoint_path, "trainer_state.json")
                    if os.path.isfile(state_file):
                        with open(state_file) as f:
                            saved_state = json.load(f)
                        initial_batch_id = saved_state.get("global_step", 0)
                        self.logger.info(
                            f"Resuming from {checkpoint_path} at step {initial_batch_id}"
                        )
                    self._resume_checkpoint_path = checkpoint_path

            # On resume with LoRA: load adapter from checkpoint and sync to vLLM
            # so the first batch is generated with the correct weights.
            if self._resume_checkpoint_path and is_peft_model(self.model):
                from peft import set_peft_model_state_dict
                from peft.utils import load_peft_weights

                adapter_weights = load_peft_weights(self._resume_checkpoint_path)
                set_peft_model_state_dict(self.model, adapter_weights)
                self.logger.info(
                    f"Loaded adapter weights from {self._resume_checkpoint_path}"
                )
                self.model.merge_adapter()
                for name, param in self.model.named_parameters():
                    name = name.removeprefix("base_model.model.").replace(
                        ".base_layer", ""
                    )
                    if self.model.prefix in name:
                        continue
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")
                    self.client.update_named_param(name, param.data)
                self.model.unmerge_adapter()
                self.client.reset_prefix_cache()
                while self.client.get_num_background_tasks() > 0:
                    time.sleep(0.5)
                self.logger.info("Synced resumed adapter weights to vLLM")

            self.orchestrator.submit_batch(initial_batch_id)
        else:
            self.orchestrator = None
            self.client = None
            self.vllm_base_url = None
            self._resume_checkpoint_path = None
            if args.resume_from_checkpoint:
                checkpoint_path = args.resume_from_checkpoint
                if checkpoint_path == "auto":
                    from transformers.trainer_utils import get_last_checkpoint

                    checkpoint_path = get_last_checkpoint(args.output_dir)
                self._resume_checkpoint_path = checkpoint_path

        # on-policy sync callback
        if self.on_policy_sync:
            self.add_callback(OnPolicySyncCallback(self))
            self.logger.info(
                "On-policy sync enabled: weight sync and batch submission "
                "will happen after optimizer.step() via callback."
            )

        # metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._textual_logs = {
            "prompt": deque(),
            "completion": deque(),
            "rewards": defaultdict(lambda: deque()),
        }

        # fixed rollouts for perplexity tracking
        self.fixed_rollout_input_ids: list | None = None
        self.fixed_rollout_completion_masks: list | None = None
        if args.rollout_perplexity_path is not None:
            self._load_fixed_rollouts(args.rollout_perplexity_path)

    def _load_fixed_rollouts(self, path: str) -> None:
        import json

        tokenizer = self.processing_class
        all_input_ids, all_completion_masks = [], []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                prompt = ex["prompt"]
                completion = ex["completion"]

                if isinstance(completion, str):
                    completion_messages = [{"role": "assistant", "content": completion}]
                else:
                    completion_messages = list(completion)

                if not isinstance(prompt, list):
                    self.logger.warning(
                        "Skipping rollout: 'prompt' must be a list of messages."
                    )
                    continue

                try:
                    full_ids = tokenizer.apply_chat_template(
                        list(prompt) + completion_messages,
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                    prompt_ids = tokenizer.apply_chat_template(
                        list(prompt),
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Skipping rollout due to tokenization error: {e}"
                    )
                    continue

                completion_len = len(full_ids) - len(prompt_ids)
                if completion_len <= 0:
                    continue

                # logprob[t] predicts token[t+1]; completion starts at prompt_ids index
                logprob_mask = [0] * (len(prompt_ids) - 1) + [1] * completion_len
                full_ids = full_ids[: self.max_seq_len]
                logprob_mask = logprob_mask[: self.max_seq_len - 1]
                all_input_ids.append(full_ids)
                all_completion_masks.append(logprob_mask)

        if not all_input_ids:
            self.logger.warning(f"No valid rollouts loaded from {path}")
            return

        self.fixed_rollout_input_ids = all_input_ids
        self.fixed_rollout_completion_masks = all_completion_masks
        self.logger.info(
            f"Loaded {len(all_input_ids)} fixed rollouts from {path} for perplexity logging."
        )

    def compute_fixed_rollout_perplexity(self) -> float | None:
        if self.fixed_rollout_input_ids is None:
            return None

        device = self.accelerator.device
        pad_id = self.processing_class.pad_token_id

        max_len = max(len(ids) for ids in self.fixed_rollout_input_ids)
        padded_ids, padded_masks = [], []
        for ids, mask in zip(
            self.fixed_rollout_input_ids, self.fixed_rollout_completion_masks
        ):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_masks.append(mask + [0] * (max_len - 1 - len(mask)))

        input_ids = torch.tensor(padded_ids, device=device)
        attn_mask = input_ids.ne(pad_id).int()
        comp_mask = torch.tensor(padded_masks, device=device, dtype=torch.float)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            logprobs, _, _ = self.get_logprobs(
                self.model, input_ids, attn_mask, batch_size=self.args.micro_batch_size
            )
        if was_training:
            self.model.train()

        num_comp_tokens = comp_mask.sum(dim=-1)
        valid = num_comp_tokens > 0
        if not valid.any():
            return None

        mean_logprob = (logprobs * comp_mask).sum(dim=-1)[valid] / num_comp_tokens[
            valid
        ]
        return torch.exp(-mean_logprob).mean().item()

    def training_step(
        self,
        model: nn.Module,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if not self.on_policy_sync:
            # Default (off-policy) path: sync weights and pre-submit next batch
            # before consuming the current batch.
            self.update_vllm()
            if self.orchestrator:
                self.orchestrator.submit_batch(self.state.global_step + 1)
        # On-policy path: sync + submit are handled by OnPolicySyncCallback
        # after optimizer.step(), so we just consume the batch here.

        broadcast_list = [None]
        if self.orchestrator:
            broadcast_list = [self.orchestrator.get_batch(self.state.global_step)]
        broadcast_object_list(broadcast_list)
        assert broadcast_list[0] is not None
        batch = broadcast_list[0]

        model.train()
        total_loss = torch.zeros((), device=self.accelerator.device)
        local_microbatches = batch.microbatches[self.accelerator.process_index]

        if batch.global_item_count <= 0:
            return total_loss

        world_size = max(self.accelerator.num_processes, 1)
        device = self.accelerator.device

        # Normalizer computed in orchestrator based on effective loss averaging level
        norm_value = torch.tensor(
            float(batch.loss_normalizer) / float(world_size),
            device=device,
            dtype=torch.float32,
        ).clamp(min=1.0)

        # With mini-batching, each optimizer step sees ~1/num_mini_batches of the tokens.
        # For global-level normalize_by_kept_tokens, the loss already encodes 1/kept_tokens
        # so we skip the outer norm_value and only apply the mini-batch scaling.
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
        _mean_level = self.loss_mean_level or _DEFAULT_MEAN_LEVELS.get(
            self.loss_type, "seq"
        )
        if self.normalize_by_kept_tokens and _mean_level == "global":
            inv_tokens_per_rank = torch.tensor(
                float(self.num_mini_batches), device=device, dtype=torch.float32
            )
        else:
            inv_tokens_per_rank = norm_value.reciprocal() * self.num_mini_batches

        ir_tracker = init_stat_tracker(device)
        entropy_tracker = init_stat_tracker(device)
        mismatch_kl_tracker = init_stat_tracker(device)
        tv_distance_tracker = init_stat_tracker(device)
        chi_squared_tracker = init_stat_tracker(device)
        perplexity_mean_tracker = init_stat_tracker(device)
        prob_train_mean_tracker = init_stat_tracker(self.accelerator.device)
        prob_train_min_tracker = init_stat_tracker(self.accelerator.device)
        prob_train_max_tracker = init_stat_tracker(self.accelerator.device)
        entropy_pos_tracker = init_stat_tracker(self.accelerator.device)
        entropy_neg_tracker = init_stat_tracker(self.accelerator.device)
        perplexity_pos_tracker = init_stat_tracker(self.accelerator.device)
        perplexity_neg_tracker = init_stat_tracker(self.accelerator.device)
        weighted_prob_pos_tracker = init_stat_tracker(self.accelerator.device)
        weighted_prob_neg_tracker = init_stat_tracker(self.accelerator.device)
        grad_mass_trackers: dict[str, dict[str, torch.Tensor]] = {}
        token_count_accumulators: dict[str, int] = {}
        device = self.accelerator.device
        pad_token_id = getattr(self.processing_class, "pad_token_id", None)
        assert pad_token_id is not None

        # Collect plot data and clipping stats across microbatches
        all_plot_data: list[dict] = []
        total_clipped_tokens = 0
        total_kept_tokens_acc = 0
        total_clipped_pos = 0
        total_kept_pos = 0
        total_clipped_neg = 0
        total_kept_neg = 0
        accumulated_bin_counts: dict[str, int] = {}

        # Split microbatches into mini-batch groups for the inner optimization loop.
        # Each mini-batch gets its own optimizer step; the last one is handled by HF Trainer.
        mbs_per_mini = len(local_microbatches) // self.num_mini_batches

        for mini_idx in range(self.num_mini_batches):
            mb_start = mini_idx * mbs_per_mini
            mb_end = mb_start + mbs_per_mini

            for microbatch in local_microbatches[mb_start:mb_end]:
                input_ids = pad(
                    [torch.tensor(x, device=device) for x in microbatch.input_ids],
                    padding_value=pad_token_id,  # type: ignore :(
                    padding_side="right",
                )
                loss_mask = pad(
                    [torch.tensor(x, device=device) for x in microbatch.loss_mask],
                    padding_side="right",
                )
                inference_logprobs = pad(
                    [
                        torch.tensor(x, device=device)
                        for x in microbatch.sampling_logprobs
                    ],
                    padding_value=0,
                    padding_side="right",
                )
                advantages = pad(
                    [torch.tensor(x, device=device) for x in microbatch.advantages],
                    padding_value=0,
                    padding_side="right",
                )
                attn_mask = input_ids.ne(pad_token_id).int()
                trainer_logprobs, entropies, sum_p_sq = self.get_logprobs(
                    model,
                    input_ids,
                    attn_mask,
                    batch_size=self.args.micro_batch_size,
                )
                loss_mask = loss_mask[:, 1:]
                inference_logprobs = inference_logprobs[:, 1:]
                advantages = advantages[:, 1:]

                seq_loss_weights = torch.tensor(
                    microbatch.seq_loss_weights, device=device
                )
                group_ids = torch.tensor(
                    microbatch.group_ids, device=device, dtype=torch.long
                )
                mb_inputs = {
                    "loss_mask": loss_mask,
                    "inference_logprobs": inference_logprobs,
                    "trainer_logprobs": trainer_logprobs,
                    "entropies": entropies,
                    "advantages": advantages,
                    "seq_loss_weights": seq_loss_weights,
                    "sum_p_sq": sum_p_sq,
                    "group_ids": group_ids,
                }
                with self.compute_loss_context_manager():
                    loss, summaries = self.compute_loss(
                        model,
                        mb_inputs,
                        num_items_in_batch=torch.tensor(self.batch_size, device=device),
                        return_outputs=True,
                    )
                self.accelerator.backward(loss * inv_tokens_per_rank)
                total_loss = total_loss + (loss.detach() * inv_tokens_per_rank)
                assert isinstance(summaries, dict)
                update_stat_tracker(ir_tracker, summaries["importance_sampling"])
                update_stat_tracker(entropy_tracker, summaries["entropy"])
                update_stat_tracker(mismatch_kl_tracker, summaries["mismatch_kl"])
                update_stat_tracker(tv_distance_tracker, summaries["tv_distance"])
                update_stat_tracker(chi_squared_tracker, summaries["chi_squared"])
                update_stat_tracker(
                    perplexity_mean_tracker, summaries["perplexity_mean"]
                )
                update_stat_tracker(
                    prob_train_mean_tracker, summaries["prob_train_mean"]
                )
                update_stat_tracker(prob_train_min_tracker, summaries["prob_train_min"])
                update_stat_tracker(prob_train_max_tracker, summaries["prob_train_max"])
                update_stat_tracker(entropy_pos_tracker, summaries["entropy_pos"])
                update_stat_tracker(entropy_neg_tracker, summaries["entropy_neg"])
                update_stat_tracker(perplexity_pos_tracker, summaries["perplexity_pos"])
                update_stat_tracker(perplexity_neg_tracker, summaries["perplexity_neg"])
                update_stat_tracker(
                    weighted_prob_pos_tracker, summaries["weighted_prob_pos"]
                )
                update_stat_tracker(
                    weighted_prob_neg_tracker, summaries["weighted_prob_neg"]
                )
                total_clipped_tokens += summaries["clipped_tokens"]
                total_kept_tokens_acc += summaries["total_kept_tokens"]
                total_clipped_pos += summaries["clipped_tokens_pos"]
                total_kept_pos += summaries["kept_tokens_pos"]
                total_clipped_neg += summaries["clipped_tokens_neg"]
                total_kept_neg += summaries["kept_tokens_neg"]
                for k, v in summaries["bin_counts"].items():
                    accumulated_bin_counts[k] = accumulated_bin_counts.get(k, 0) + v
                for gm_key, gm_summary in summaries["grad_mass"].items():
                    if gm_key not in grad_mass_trackers:
                        grad_mass_trackers[gm_key] = init_stat_tracker(device)
                    update_stat_tracker(grad_mass_trackers[gm_key], gm_summary)
                for tc_key, tc_val in summaries["token_counts"].items():
                    token_count_accumulators[tc_key] = (
                        token_count_accumulators.get(tc_key, 0) + tc_val
                    )
                all_plot_data.append(summaries["plot_data"])

            # For all mini-batches except the last: clip gradients, step optimizer, zero grad.
            # The last mini-batch is handled by HF Trainer after training_step returns.
            if mini_idx < self.num_mini_batches - 1:
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        model.parameters(), self.args.max_grad_norm
                    )
                self.optimizer.step()
                model.zero_grad()

        # Save probability scatter plot
        if all_plot_data:
            import numpy as np

            prob_inference_all = np.concatenate(
                [d["prob_inference"] for d in all_plot_data]
            )
            prob_train_all = np.concatenate([d["prob_train"] for d in all_plot_data])
            advantages_all = np.concatenate([d["advantages"] for d in all_plot_data])

            # Sample if too many points
            max_points = 10000
            if len(prob_inference_all) > max_points:
                indices = np.random.choice(
                    len(prob_inference_all), max_points, replace=False
                )
                prob_inference_all = prob_inference_all[indices]
                prob_train_all = prob_train_all[indices]
                advantages_all = advantages_all[indices]

            colors = ["green" if adv >= 0 else "red" for adv in advantages_all]

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(prob_inference_all, prob_train_all, c=colors, alpha=0.5, s=10)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)  # diagonal line
            ax.set_xlabel("Inference Probability")
            ax.set_ylabel("Train Probability")
            ax.set_title(f"Step {self.state.global_step}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            plots_dir = self.args.output_dir.replace("checkpoints", "plots")
            os.makedirs(plots_dir, exist_ok=True)
            fig.savefig(
                os.path.join(plots_dir, f"step_{self.state.global_step}.png"), dpi=100
            )
            plt.close(fig)

            # Save scatter plot data
            data_dir = os.path.join(plots_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(data_dir, f"scatter_step_{self.state.global_step}.npz"),
                prob_inference=prob_inference_all,
                prob_train=prob_train_all,
                advantages=advantages_all,
            )

            # Save train probability distribution split by advantage sign
            pos_mask = advantages_all >= 0
            neg_mask = advantages_all < 0
            prob_train_pos = prob_train_all[pos_mask]
            prob_train_neg = prob_train_all[neg_mask]

            fig, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=(14, 5))
            bins = np.linspace(0, 1, 51)

            ax_pos.hist(
                prob_train_pos, bins=bins, color="green", alpha=0.7, edgecolor="black"
            )
            ax_pos.set_xlabel("Train Probability")
            ax_pos.set_ylabel("Count")
            ax_pos.set_title(f"Positive Advantage Tokens (n={len(prob_train_pos)})")
            ax_pos.set_xlim(0, 1)

            ax_neg.hist(
                prob_train_neg, bins=bins, color="red", alpha=0.7, edgecolor="black"
            )
            ax_neg.set_xlabel("Train Probability")
            ax_neg.set_ylabel("Count")
            ax_neg.set_title(f"Negative Advantage Tokens (n={len(prob_train_neg)})")
            ax_neg.set_xlim(0, 1)

            fig.suptitle(
                f"Train Probability Distribution - Step {self.state.global_step}"
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(plots_dir, f"prob_dist_step_{self.state.global_step}.png"),
                dpi=100,
            )
            plt.close(fig)

            # Save prob distribution data
            np.savez_compressed(
                os.path.join(data_dir, f"prob_dist_step_{self.state.global_step}.npz"),
                prob_train_pos=prob_train_pos,
                prob_train_neg=prob_train_neg,
            )

        # Save reward distribution histograms
        if self.accelerator.is_main_process and batch.rewards_dict:
            import numpy as np

            plots_dir = self.args.output_dir.replace("checkpoints", "plots")
            rewards_plot_dir = os.path.join(plots_dir, "rewards")
            rewards_data_dir = os.path.join(plots_dir, "data", "rewards")
            os.makedirs(rewards_plot_dir, exist_ok=True)
            os.makedirs(rewards_data_dir, exist_ok=True)
            for rname, rvalues in batch.rewards_dict.items():
                if not rvalues:
                    continue
                rvalues_arr = np.asarray(rvalues, dtype=np.float32)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(rvalues_arr, bins=20, edgecolor="black", alpha=0.7)
                ax.set_xlabel(rname)
                ax.set_ylabel("Number of Rollouts")
                ax.set_title(f"{rname} Distribution - Step {self.state.global_step}")
                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        rewards_plot_dir,
                        f"{rname}_step_{self.state.global_step}.png",
                    ),
                    dpi=100,
                )
                plt.close(fig)
                np.save(
                    os.path.join(
                        rewards_data_dir,
                        f"{rname}_step_{self.state.global_step}.npy",
                    ),
                    rvalues_arr,
                )

        # Token taxonomy bin counts: save data + plot
        if accumulated_bin_counts and self.accelerator.is_main_process:
            _BIN_TYPES = ("valley", "peak")
            _SIGNS = ("pos", "neg")
            step = self.state.global_step
            plots_dir = self.args.output_dir.replace("checkpoints", "plots")
            bin_data_dir = os.path.join(plots_dir, "data", "token_bins")
            bin_plot_dir = os.path.join(plots_dir, "token_bins")
            os.makedirs(bin_data_dir, exist_ok=True)
            os.makedirs(bin_plot_dir, exist_ok=True)

            hyperparams = {
                "include_bins": self.include_bins,
                "exclude_bins": self.exclude_bins,
            }

            # Append step record to JSONL
            record = {
                "step": step,
                "hyperparams": hyperparams,
                "counts": accumulated_bin_counts,
            }
            jsonl_path = os.path.join(bin_data_dir, "token_bins.jsonl")
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            # Bar chart: eligible vs kept per bin, pos and neg side by side
            bin_labels = list(_BIN_TYPES)
            colors = ["#FF9800", "#2196F3"]  # valley=orange, peak=blue
            bar_w = 0.35
            x = np.arange(len(bin_labels))

            fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
            for ax, sign in zip(axes[0], _SIGNS):
                eligible = [
                    accumulated_bin_counts.get(f"{sign}_{t}_eligible", 0)
                    for t in _BIN_TYPES
                ]
                kept = [
                    accumulated_bin_counts.get(f"{sign}_{t}_kept", 0)
                    for t in _BIN_TYPES
                ]
                ax.bar(
                    x - bar_w / 2,
                    eligible,
                    bar_w,
                    label="eligible",
                    color=colors,
                    alpha=0.6,
                )
                ax.bar(
                    x + bar_w / 2, kept, bar_w, label="kept", color=colors, alpha=1.0
                )
                ax.set_xticks(x)
                ax.set_xticklabels(bin_labels)
                ax.set_ylabel("Token count")
                ax.set_title(f"{'Positive' if sign == 'pos' else 'Negative'} advantage")
                ax.legend()

            fig.suptitle(f"Token Bins (valley: p<Σp², peak: p≥Σp²) — Step {step}")
            fig.tight_layout()
            fig.savefig(os.path.join(bin_plot_dir, f"bins_step_{step}.png"), dpi=100)
            plt.close(fig)

        ir_mean = finalize_stat_tracker(ir_tracker, self.accelerator)
        entropy_mean = finalize_stat_tracker(entropy_tracker, self.accelerator)
        mismatch_kl_mean = finalize_stat_tracker(mismatch_kl_tracker, self.accelerator)
        tv_distance_mean = finalize_stat_tracker(tv_distance_tracker, self.accelerator)
        chi_squared_mean = finalize_stat_tracker(chi_squared_tracker, self.accelerator)
        perplexity_mean = finalize_stat_tracker(
            perplexity_mean_tracker, self.accelerator
        )
        prob_train_mean = finalize_stat_tracker(
            prob_train_mean_tracker, self.accelerator
        )
        prob_train_min = finalize_stat_tracker(prob_train_min_tracker, self.accelerator)
        prob_train_max = finalize_stat_tracker(prob_train_max_tracker, self.accelerator)
        entropy_pos_mean = finalize_stat_tracker(entropy_pos_tracker, self.accelerator)
        entropy_neg_mean = finalize_stat_tracker(entropy_neg_tracker, self.accelerator)
        perplexity_pos_mean = finalize_stat_tracker(
            perplexity_pos_tracker, self.accelerator
        )
        perplexity_neg_mean = finalize_stat_tracker(
            perplexity_neg_tracker, self.accelerator
        )
        weighted_prob_pos_mean = finalize_stat_tracker(
            weighted_prob_pos_tracker, self.accelerator
        )
        weighted_prob_neg_mean = finalize_stat_tracker(
            weighted_prob_neg_tracker, self.accelerator
        )
        assert ir_mean is not None
        assert entropy_mean is not None
        assert mismatch_kl_mean is not None

        extra_metrics: dict[str, float] = {
            "importance_ratio": ir_mean,
            "entropy": entropy_mean,
            "mismatch_kl": mismatch_kl_mean,
        }
        if tv_distance_mean is not None:
            extra_metrics["tv_distance"] = tv_distance_mean
        if chi_squared_mean is not None:
            extra_metrics["chi_squared"] = chi_squared_mean
        if perplexity_mean is not None:
            extra_metrics["perplexity_mean"] = perplexity_mean
        if prob_train_mean is not None:
            extra_metrics["prob_train_mean"] = prob_train_mean
        if prob_train_min is not None:
            extra_metrics["prob_train_min"] = prob_train_min
        if prob_train_max is not None:
            extra_metrics["prob_train_max"] = prob_train_max
        if entropy_pos_mean is not None:
            extra_metrics["entropy_pos"] = entropy_pos_mean
        if entropy_neg_mean is not None:
            extra_metrics["entropy_neg"] = entropy_neg_mean
        if perplexity_pos_mean is not None:
            extra_metrics["perplexity_pos"] = perplexity_pos_mean
        if perplexity_neg_mean is not None:
            extra_metrics["perplexity_neg"] = perplexity_neg_mean
        if weighted_prob_pos_mean is not None:
            extra_metrics["weighted_prob_pos"] = weighted_prob_pos_mean
        if weighted_prob_neg_mean is not None:
            extra_metrics["weighted_prob_neg"] = weighted_prob_neg_mean
        if total_kept_tokens_acc > 0:
            extra_metrics["clipped_fraction"] = (
                total_clipped_tokens / total_kept_tokens_acc
            )
            extra_metrics["clipped_tokens"] = total_clipped_tokens
        if total_kept_pos > 0:
            extra_metrics["clipped_fraction_pos"] = total_clipped_pos / total_kept_pos
        if total_kept_neg > 0:
            extra_metrics["clipped_fraction_neg"] = total_clipped_neg / total_kept_neg
        gm_values: dict[str, float] = {}
        for gm_key, gm_tracker in grad_mass_trackers.items():
            gm_mean = finalize_stat_tracker(gm_tracker, self.accelerator)
            if gm_mean is not None:
                gm_values[gm_key] = gm_mean

        # Save grad_mass and token_count bar charts per probability bin
        prob_bin_labels = ["0-0.01", "0.01-0.1", "0.1-0.25", "0.25-0.5", "0.5-1.0"]
        plots_dir = self.args.output_dir.replace("checkpoints", "plots")

        # Grad mass plot
        gm_pos = [gm_values.get(f"grad_mass/{b}/pos", 0.0) for b in prob_bin_labels]
        gm_neg = [gm_values.get(f"grad_mass/{b}/neg", 0.0) for b in prob_bin_labels]
        if any(v != 0 for v in gm_pos + gm_neg):
            gm_plot_dir = os.path.join(plots_dir, "grad_mass")
            gm_data_dir = os.path.join(plots_dir, "data", "grad_mass")
            os.makedirs(gm_plot_dir, exist_ok=True)
            os.makedirs(gm_data_dir, exist_ok=True)
            fig, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=(14, 5))
            x = range(len(prob_bin_labels))
            ax_pos.bar(x, gm_pos, color="green", alpha=0.7, edgecolor="black")
            ax_pos.set_xticks(x)
            ax_pos.set_xticklabels(prob_bin_labels, rotation=45)
            ax_pos.set_xlabel("Probability Bin")
            ax_pos.set_ylabel("Mean Grad Mass")
            ax_pos.set_title("Positive Advantage")
            ax_neg.bar(x, gm_neg, color="red", alpha=0.7, edgecolor="black")
            ax_neg.set_xticks(x)
            ax_neg.set_xticklabels(prob_bin_labels, rotation=45)
            ax_neg.set_xlabel("Probability Bin")
            ax_neg.set_ylabel("Mean Grad Mass")
            ax_neg.set_title("Negative Advantage")
            fig.suptitle(f"Grad Mass by Prob Bin - Step {self.state.global_step}")
            fig.tight_layout()
            fig.savefig(
                os.path.join(gm_plot_dir, f"step_{self.state.global_step}.png"), dpi=100
            )
            plt.close(fig)
            np.savez_compressed(
                os.path.join(gm_data_dir, f"step_{self.state.global_step}.npz"),
                bins=np.array(prob_bin_labels),
                pos=np.array(gm_pos),
                neg=np.array(gm_neg),
            )

        # Token count plot
        tc_pos = [
            token_count_accumulators.get(f"token_count/{b}/pos", 0)
            for b in prob_bin_labels
        ]
        tc_neg = [
            token_count_accumulators.get(f"token_count/{b}/neg", 0)
            for b in prob_bin_labels
        ]
        if any(v != 0 for v in tc_pos + tc_neg):
            tc_plot_dir = os.path.join(plots_dir, "token_count")
            tc_data_dir = os.path.join(plots_dir, "data", "token_count")
            os.makedirs(tc_plot_dir, exist_ok=True)
            os.makedirs(tc_data_dir, exist_ok=True)
            fig, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=(14, 5))
            x = range(len(prob_bin_labels))
            ax_pos.bar(x, tc_pos, color="green", alpha=0.7, edgecolor="black")
            ax_pos.set_xticks(x)
            ax_pos.set_xticklabels(prob_bin_labels, rotation=45)
            ax_pos.set_xlabel("Probability Bin")
            ax_pos.set_ylabel("Token Count")
            ax_pos.set_title("Positive Advantage")
            ax_neg.bar(x, tc_neg, color="red", alpha=0.7, edgecolor="black")
            ax_neg.set_xticks(x)
            ax_neg.set_xticklabels(prob_bin_labels, rotation=45)
            ax_neg.set_xlabel("Probability Bin")
            ax_neg.set_ylabel("Token Count")
            ax_neg.set_title("Negative Advantage")
            fig.suptitle(f"Token Count by Prob Bin - Step {self.state.global_step}")
            fig.tight_layout()
            fig.savefig(
                os.path.join(tc_plot_dir, f"step_{self.state.global_step}.png"), dpi=100
            )
            plt.close(fig)
            np.savez_compressed(
                os.path.join(tc_data_dir, f"step_{self.state.global_step}.npz"),
                bins=np.array(prob_bin_labels),
                pos=np.array(tc_pos),
                neg=np.array(tc_neg),
            )

        # Token strength plots: IR * adv * scaling_factor per probability bin
        if all_plot_data and "token_strength" in all_plot_data[0]:
            import numpy as np

            ts_all = np.concatenate([d["token_strength"] for d in all_plot_data])
            pt_all = np.concatenate([d["prob_train"] for d in all_plot_data])
            sa_all = np.concatenate([d["seq_advantage_sign"] for d in all_plot_data])
            pos_idx = sa_all > 0
            neg_idx = sa_all < 0

            ts_plot_dir = os.path.join(plots_dir, "token_strength")
            ts_data_dir = os.path.join(plots_dir, "data", "token_strength")
            os.makedirs(ts_plot_dir, exist_ok=True)
            os.makedirs(ts_data_dir, exist_ok=True)

            # --- Coarse bins (same as grad_mass) ---
            coarse_bins = [(0, 0.01), (0.01, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)]
            coarse_labels = [f"{lo}-{hi}" for lo, hi in coarse_bins]

            def _bin_mean(vals, probs, mask, bins):
                means = []
                for lo, hi in bins:
                    in_bin = (probs >= lo) & (probs < hi) & mask
                    means.append(float(vals[in_bin].mean()) if in_bin.any() else 0.0)
                return means

            ts_pos_coarse = _bin_mean(ts_all, pt_all, pos_idx, coarse_bins)
            ts_neg_coarse = _bin_mean(ts_all, pt_all, neg_idx, coarse_bins)

            fig, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=(14, 5))
            x = range(len(coarse_labels))
            ax_pos.bar(x, ts_pos_coarse, color="green", alpha=0.7, edgecolor="black")
            ax_pos.set_xticks(x)
            ax_pos.set_xticklabels(coarse_labels, rotation=45)
            ax_pos.set_xlabel("Probability Bin")
            ax_pos.set_ylabel("Mean Token Strength")
            ax_pos.set_title("Positive Advantage")
            ax_neg.bar(x, ts_neg_coarse, color="red", alpha=0.7, edgecolor="black")
            ax_neg.set_xticks(x)
            ax_neg.set_xticklabels(coarse_labels, rotation=45)
            ax_neg.set_xlabel("Probability Bin")
            ax_neg.set_ylabel("Mean Token Strength")
            ax_neg.set_title("Negative Advantage")
            fig.suptitle(f"Token Strength (coarse) - Step {self.state.global_step}")
            fig.tight_layout()
            fig.savefig(
                os.path.join(ts_plot_dir, f"coarse_step_{self.state.global_step}.png"),
                dpi=100,
            )
            plt.close(fig)
            np.savez_compressed(
                os.path.join(ts_data_dir, f"coarse_step_{self.state.global_step}.npz"),
                bins=np.array(coarse_labels),
                pos=np.array(ts_pos_coarse),
                neg=np.array(ts_neg_coarse),
            )

            # --- Fine bins (50 uniform bins across 0-1) ---
            fine_edges = np.linspace(0, 1, 51)
            fine_bins = [(fine_edges[i], fine_edges[i + 1]) for i in range(50)]
            fine_labels = [f"{lo:.2f}" for lo, _ in fine_bins]

            ts_pos_fine = _bin_mean(ts_all, pt_all, pos_idx, fine_bins)
            ts_neg_fine = _bin_mean(ts_all, pt_all, neg_idx, fine_bins)

            fig, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=(16, 5))
            x = range(len(fine_labels))
            ax_pos.bar(x, ts_pos_fine, color="green", alpha=0.7)
            ax_pos.set_xticks(x[::5])
            ax_pos.set_xticklabels([fine_labels[i] for i in x[::5]], rotation=45)
            ax_pos.set_xlabel("Probability Bin")
            ax_pos.set_ylabel("Mean Token Strength")
            ax_pos.set_title("Positive Advantage")
            ax_neg.bar(x, ts_neg_fine, color="red", alpha=0.7)
            ax_neg.set_xticks(x[::5])
            ax_neg.set_xticklabels([fine_labels[i] for i in x[::5]], rotation=45)
            ax_neg.set_xlabel("Probability Bin")
            ax_neg.set_ylabel("Mean Token Strength")
            ax_neg.set_title("Negative Advantage")
            fig.suptitle(f"Token Strength (fine) - Step {self.state.global_step}")
            fig.tight_layout()
            fig.savefig(
                os.path.join(ts_plot_dir, f"fine_step_{self.state.global_step}.png"),
                dpi=100,
            )
            plt.close(fig)
            np.savez_compressed(
                os.path.join(ts_data_dir, f"fine_step_{self.state.global_step}.npz"),
                bins=np.array(fine_labels),
                pos=np.array(ts_pos_fine),
                neg=np.array(ts_neg_fine),
            )

        if self.accelerator.is_main_process:
            metrics_to_log = {**batch.metrics_dict, **extra_metrics}
            self.log_metrics(
                mode="train",
                batch_metrics=metrics_to_log,
            )
            self.log_rollouts(
                prompts=batch.prompts,
                completions=batch.completions,
                rewards_dict=batch.rewards_dict,
            )

        self.maybe_clear_cache()
        return total_loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        loss_mask = inputs["loss_mask"].bool()
        entropies = inputs["entropies"]
        trainer_logprobs = inputs["trainer_logprobs"]
        inference_logprobs = inputs["inference_logprobs"]
        advantages = inputs["advantages"]
        seq_loss_weights = inputs["seq_loss_weights"]

        token_num_sum = inputs["loss_mask"].sum(dim=1)
        # Calculate importance ratio at sequence level
        log_importance_ratio = trainer_logprobs - inference_logprobs
        log_seq_ir_sum = (log_importance_ratio * inputs["loss_mask"]).sum(dim=1)
        log_seq_ir = log_seq_ir_sum / (token_num_sum + 1e-8)
        sequence_ir = torch.exp(log_seq_ir)
        sequence_ir = sequence_ir.unsqueeze(1)
        token_ir = torch.exp(log_importance_ratio)

        keep_mask = loss_mask

        # Sequence-level masking for non-positive advantages
        if self.mask_negative_advantages:
            seq_advantage = (advantages * inputs["loss_mask"]).sum(dim=1) / (
                token_num_sum + 1e-8
            )
            non_pos_adv_mask = (seq_advantage <= 0).unsqueeze(1)
            keep_mask = keep_mask & ~non_pos_adv_mask

        # Sequence-level masking for non-negative advantages
        if self.mask_positive_advantages:
            seq_advantage = (advantages * inputs["loss_mask"]).sum(dim=1) / (
                token_num_sum + 1e-8
            )
            non_neg_adv_mask = (seq_advantage >= 0).unsqueeze(1)
            keep_mask = keep_mask & ~non_neg_adv_mask

        # Token taxonomy: valley (p_train < sum_p_sq) vs peak (p_train >= sum_p_sq)
        _BIN_TYPES = ("valley", "peak")
        _SIGNS = ("pos", "neg")
        _ALL_BINS = frozenset(f"{s}_{t}" for s in _SIGNS for t in _BIN_TYPES)

        def _expand_bin_spec(specs):
            out = set()
            for s in specs:
                if s in _ALL_BINS:
                    out.add(s)
                elif s in _BIN_TYPES:
                    out.update(f"{sign}_{s}" for sign in _SIGNS)
                elif s in _SIGNS:
                    out.update(f"{s}_{t}" for t in _BIN_TYPES)
            return out

        sum_p_sq = inputs.get("sum_p_sq")
        taxonomy_enabled = self.token_bin_counting
        bin_counts: dict[str, int] = {}
        if taxonomy_enabled:
            lm = loss_mask
            p_train = torch.exp(trainer_logprobs)

            if sum_p_sq is not None:
                is_valley = (p_train < sum_p_sq) & lm
            else:
                is_valley = torch.zeros_like(lm)
            is_peak = lm & ~is_valley

            type_bins = {
                "valley": is_valley,
                "peak": is_peak,
            }

            seq_advantage = (advantages * lm).sum(dim=1) / (token_num_sum + 1e-8)
            pos_seq = (seq_advantage >= 0).unsqueeze(1)
            neg_seq = (seq_advantage < 0).unsqueeze(1)
            sign_masks = {"pos": pos_seq, "neg": neg_seq}

            # Apply include/exclude masking
            if self.include_bins is not None or self.exclude_bins is not None:
                if self.include_bins is not None:
                    active = _expand_bin_spec(self.include_bins)
                else:
                    active = set(_ALL_BINS) - _expand_bin_spec(self.exclude_bins or [])
                bin_keep = torch.zeros_like(keep_mask)
                for sign, smask in sign_masks.items():
                    for btype, bmask in type_bins.items():
                        if f"{sign}_{btype}" in active:
                            bin_keep = bin_keep | (smask & bmask)
                keep_mask = keep_mask & bin_keep

            with torch.no_grad():
                for sign, smask in sign_masks.items():
                    for btype, bmask in type_bins.items():
                        key = f"{sign}_{btype}"
                        bin_counts[f"{key}_eligible"] = (smask & bmask).sum().item()
                        bin_counts[f"{key}_kept"] = (
                            (smask & bmask & keep_mask).sum().item()
                        )

        # Renormalize seq_loss_weights by kept-token counts after all masking
        if self.normalize_by_kept_tokens:
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
            mean_level = self.loss_mean_level or _DEFAULT_MEAN_LEVELS.get(
                self.loss_type, "seq"
            )
            with torch.no_grad():
                kept_per_seq = keep_mask.sum(dim=1).float().clamp(min=1)  # (B,)
                if mean_level == "seq":
                    seq_loss_weights = 1.0 / kept_per_seq
                elif mean_level == "group":
                    group_ids = inputs["group_ids"]  # (B,)
                    new_weights = torch.zeros_like(kept_per_seq)
                    for g in group_ids.unique():
                        g_mask = group_ids == g
                        new_weights[g_mask] = 1.0 / kept_per_seq[g_mask].sum().clamp(
                            min=1
                        )
                    seq_loss_weights = new_weights
                else:  # global
                    seq_loss_weights = torch.ones_like(
                        kept_per_seq
                    ) / keep_mask.sum().float().clamp(min=1)

        # Initialize for clipped token counting
        raw_ir_for_count = None
        clipped_ir_for_count = None

        # Compute loss based on loss_type
        if self.loss_type == "grpo":
            clipped = torch.clamp(token_ir, 1 - self.clip_eps, 1 + self.clip_eps)
            per_token_loss = -torch.min(token_ir * advantages, clipped * advantages)
            raw_ir_for_count, clipped_ir_for_count = token_ir, clipped
            loss = (per_token_loss * keep_mask * seq_loss_weights.unsqueeze(1)).sum()

        elif self.loss_type == "dr_grpo":
            clipped = torch.clamp(token_ir, 1 - self.clip_eps, 1 + self.clip_eps)
            per_token_loss = -torch.min(token_ir * advantages, clipped * advantages)
            raw_ir_for_count, clipped_ir_for_count = token_ir, clipped
            loss = (per_token_loss * keep_mask * seq_loss_weights.unsqueeze(1)).sum()

        elif self.loss_type in ("dapo", "dr_dapo"):
            clipped = torch.clamp(token_ir, self.mask_ratio_low, self.mask_ratio_high)
            per_token_loss = -torch.min(token_ir * advantages, clipped * advantages)
            raw_ir_for_count, clipped_ir_for_count = token_ir, clipped
            loss = (per_token_loss * keep_mask * seq_loss_weights.unsqueeze(1)).sum()

        elif self.loss_type == "dr_dapo_seq":
            clipped = torch.clamp(token_ir, self.mask_ratio_low, self.mask_ratio_high)
            per_token_loss = -torch.min(token_ir * advantages, clipped * advantages)
            raw_ir_for_count, clipped_ir_for_count = token_ir, clipped
            loss = (per_token_loss * keep_mask * seq_loss_weights.unsqueeze(1)).sum()

        elif self.loss_type == "gspo":
            clipped = torch.clamp(sequence_ir, 1 - self.clip_eps, 1 + self.clip_eps)
            per_token_loss = -torch.min(sequence_ir * advantages, clipped * advantages)
            raw_ir_for_count, clipped_ir_for_count = sequence_ir, clipped
            loss = (per_token_loss * keep_mask * seq_loss_weights.unsqueeze(1)).sum()

        elif self.loss_type == "raft":
            per_token_loss = -trainer_logprobs
            loss = (per_token_loss * keep_mask * seq_loss_weights.unsqueeze(1)).sum()

        elif self.loss_type == "raft++":
            clipped = torch.clamp(token_ir, max=1 + self.clip_eps)
            per_token_loss = -torch.min(token_ir * advantages, clipped * advantages)
            raw_ir_for_count, clipped_ir_for_count = token_ir, clipped
            loss = (per_token_loss * keep_mask * seq_loss_weights.unsqueeze(1)).sum()

        elif self.loss_type == "wapo":
            # Positive adv only; denominator=group_size*max_seq_len (set in orchestrator).
            if self.use_importance_sampling:
                clipped = torch.clamp(token_ir, max=1 + self.clip_eps)
                per_token_loss = -torch.min(token_ir * advantages, clipped * advantages)
                raw_ir_for_count, clipped_ir_for_count = token_ir, clipped
            else:
                per_token_loss = -trainer_logprobs * advantages
            loss = (per_token_loss * keep_mask * seq_loss_weights.unsqueeze(1)).sum()

        else:
            raise ValueError(
                f"Invalid loss_type: {self.loss_type}. "
                f"Expected one of: grpo, dr_grpo, dapo, dr_dapo, dr_dapo_seq, gspo, raft, raft++, wapo."
            )

        # Count clipped tokens (where clamped IR != raw IR) within keep_mask
        with torch.no_grad():
            if raw_ir_for_count is not None and clipped_ir_for_count is not None:
                is_clipped = (clipped_ir_for_count != raw_ir_for_count) & keep_mask
                clipped_tokens = is_clipped.sum().item()
                total_kept_tokens = keep_mask.sum().item()
                # Split by advantage sign
                seq_adv_sign = (advantages * inputs["loss_mask"]).sum(dim=1) / (
                    token_num_sum + 1e-8
                )
                pos_seq_mask = (seq_adv_sign > 0).unsqueeze(1) & keep_mask
                neg_seq_mask = (seq_adv_sign < 0).unsqueeze(1) & keep_mask
                clipped_tokens_pos = (is_clipped & pos_seq_mask).sum().item()
                kept_tokens_pos = pos_seq_mask.sum().item()
                clipped_tokens_neg = (is_clipped & neg_seq_mask).sum().item()
                kept_tokens_neg = neg_seq_mask.sum().item()
            else:
                clipped_tokens = 0
                total_kept_tokens = max(keep_mask.sum().item(), 1)
                clipped_tokens_pos = 0
                kept_tokens_pos = 0
                clipped_tokens_neg = 0
                kept_tokens_neg = 0
            clipped_fraction = clipped_tokens / max(total_kept_tokens, 1)

        mismatch_kl = torch.exp(log_importance_ratio) - log_importance_ratio - 1

        with torch.no_grad():
            ir_summary = summarize_values(sequence_ir)
            entropy_summary = summarize_values(entropies[loss_mask])
            mismatch_kl_summary = summarize_values(mismatch_kl[loss_mask])

            # TV distance: max over tokens of |prob(train) - prob(inference)|
            prob_train = torch.exp(trainer_logprobs)
            prob_inference = torch.exp(inference_logprobs)
            tv_distance_per_token = torch.abs(prob_train - prob_inference)
            # Get max per sequence (only over valid tokens)
            tv_masked = tv_distance_per_token.masked_fill(~loss_mask, float("-inf"))
            tv_max_per_seq = tv_masked.max(dim=1)[0]
            # Filter out sequences with no valid tokens (would be -inf)
            valid_seqs = tv_max_per_seq != float("-inf")
            tv_distance_summary = summarize_values(tv_max_per_seq[valid_seqs])

            # Chi-squared variance: Mean((prob_train/prob_inference)^2) - 1 per sequence
            importance_ratio = torch.exp(log_importance_ratio)
            ir_squared = importance_ratio**2
            # Compute mean per sequence, then subtract 1
            ir_squared_masked = ir_squared * inputs["loss_mask"]
            ir_squared_sum_per_seq = ir_squared_masked.sum(dim=1)
            token_count_per_seq = inputs["loss_mask"].sum(dim=1)
            ir_squared_mean_per_seq = ir_squared_sum_per_seq / (
                token_count_per_seq + 1e-8
            )
            chi_squared_per_seq = ir_squared_mean_per_seq - 1
            valid_chi_seqs = token_count_per_seq > 0
            chi_squared_summary = summarize_values(chi_squared_per_seq[valid_chi_seqs])

            # Perplexity: exp(-mean(logprobs)) per sequence
            trainer_logprobs_masked = trainer_logprobs * inputs["loss_mask"]
            trainer_logprobs_sum_per_seq = trainer_logprobs_masked.sum(dim=1)
            mean_logprob_per_seq = trainer_logprobs_sum_per_seq / (
                token_count_per_seq + 1e-8
            )
            perplexity_per_seq = torch.exp(-mean_logprob_per_seq)
            valid_ppl_seqs = token_count_per_seq > 0
            valid_perplexities = perplexity_per_seq[valid_ppl_seqs]
            perplexity_mean_summary = summarize_values(valid_perplexities)

            # Training probabilities statistics
            prob_train = torch.exp(trainer_logprobs)
            prob_inference = torch.exp(inference_logprobs)
            valid_prob_train = prob_train[loss_mask]
            prob_train_mean_summary = summarize_values(valid_prob_train)
            prob_train_min_summary = (
                summarize_values(valid_prob_train.min().unsqueeze(0))
                if valid_prob_train.numel() > 0
                else summarize_values(valid_prob_train)
            )
            prob_train_max_summary = (
                summarize_values(valid_prob_train.max().unsqueeze(0))
                if valid_prob_train.numel() > 0
                else summarize_values(valid_prob_train)
            )

            # Entropy split by advantage sign (per-sequence)
            # advantages has same value across tokens in a sequence; get per-seq sign
            seq_advantage = (advantages * inputs["loss_mask"]).sum(dim=1) / (
                token_count_per_seq + 1e-8
            )
            # Mean entropy per sequence
            entropy_masked = entropies * inputs["loss_mask"]
            entropy_sum_per_seq = entropy_masked.sum(dim=1)
            entropy_mean_per_seq = entropy_sum_per_seq / (token_count_per_seq + 1e-8)
            valid_entropy_seqs = token_count_per_seq > 0

            pos_mask = (seq_advantage >= 0) & valid_entropy_seqs
            neg_mask = (seq_advantage < 0) & valid_entropy_seqs
            entropy_pos_summary = summarize_values(entropy_mean_per_seq[pos_mask])
            entropy_neg_summary = summarize_values(entropy_mean_per_seq[neg_mask])
            perplexity_pos_summary = summarize_values(perplexity_per_seq[pos_mask])
            perplexity_neg_summary = summarize_values(perplexity_per_seq[neg_mask])

            # Gradient mass by probability bin, split by advantage sign
            # Gradient of loss w.r.t. logit: weight * IR * (1-p_train) * |A|
            # where weight = seq_loss_weights (encodes the per-token averaging scale)
            if self.loss_type == "gspo":
                grad_magnitude = (
                    seq_loss_weights.unsqueeze(1)
                    * sequence_ir
                    * (1 - prob_train)
                    * torch.abs(advantages)
                )
            elif self.loss_type == "raft":
                grad_magnitude = seq_loss_weights.unsqueeze(1) * (1 - prob_train)
            elif self.loss_type == "raft++":
                grad_magnitude = (
                    seq_loss_weights.unsqueeze(1)
                    * token_ir
                    * (1 - prob_train)
                    * torch.abs(advantages)
                )
            else:
                grad_magnitude = (
                    seq_loss_weights.unsqueeze(1)
                    * token_ir
                    * (1 - prob_train)
                    * torch.abs(advantages)
                )
            # Zero out tokens where clipping kills the gradient
            if raw_ir_for_count is not None and clipped_ir_for_count is not None:
                # PPO min(IR*A, clip(IR)*A): gradient is zero when clipped branch wins
                grad_active = (
                    raw_ir_for_count * advantages <= clipped_ir_for_count * advantages
                )
                grad_magnitude = grad_magnitude * grad_active.float()
            pos_seq_2d = (seq_advantage > 0).unsqueeze(1) & loss_mask
            neg_seq_2d = (seq_advantage < 0).unsqueeze(1) & loss_mask
            prob_bins = [
                (0, 0.01),
                (0.01, 0.1),
                (0.1, 0.25),
                (0.25, 0.5),
                (0.5, 1.0),
            ]
            grad_mass_summaries = {}
            token_count_summaries: dict[str, int] = {}
            for lo, hi in prob_bins:
                bin_mask = (prob_train >= lo) & (prob_train < hi) & loss_mask
                bin_label = f"{lo}-{hi}"
                for sign, sign_mask in [("pos", pos_seq_2d), ("neg", neg_seq_2d)]:
                    combined = bin_mask & sign_mask
                    vals = grad_magnitude[combined]
                    grad_mass_summaries[f"grad_mass/{bin_label}/{sign}"] = (
                        summarize_values(vals)
                    )
                    token_count_summaries[f"token_count/{bin_label}/{sign}"] = int(
                        combined.sum().item()
                    )

            # Token strength: IR * adv * scaling_factor (measures contribution to grad)
            if self.loss_type == "gspo":
                token_strength = (
                    seq_loss_weights.unsqueeze(1) * sequence_ir * advantages
                )
            elif self.loss_type == "raft":
                token_strength = seq_loss_weights.unsqueeze(1) * advantages
            else:
                token_strength = seq_loss_weights.unsqueeze(1) * token_ir * advantages

            # Weighted probability: sum(prob * |strength|) / sum(|strength|)
            # Use keep_mask so masked-out tokens don't contribute
            abs_ts = torch.abs(token_strength)
            pos_keep = (seq_advantage > 0).unsqueeze(1) & keep_mask
            neg_keep = (seq_advantage < 0).unsqueeze(1) & keep_mask
            pos_abs_ts = abs_ts * pos_keep.float()
            neg_abs_ts = abs_ts * neg_keep.float()
            weighted_prob_pos = (prob_train * pos_abs_ts).sum() / (
                pos_abs_ts.sum() + 1e-8
            )
            weighted_prob_neg = (prob_train * neg_abs_ts).sum() / (
                neg_abs_ts.sum() + 1e-8
            )
            weighted_prob_pos_summary = summarize_values(weighted_prob_pos.unsqueeze(0))
            weighted_prob_neg_summary = summarize_values(weighted_prob_neg.unsqueeze(0))

            # Collect data for plotting (sample to avoid memory issues)
            valid_prob_inference = prob_inference[loss_mask]
            valid_advantages = advantages[loss_mask]
            valid_token_strength = token_strength[loss_mask]
            valid_seq_advantage = seq_advantage.unsqueeze(1).expand_as(prob_train)[
                loss_mask
            ]
            plot_data = {
                "prob_inference": valid_prob_inference.detach().cpu().float().numpy(),
                "prob_train": valid_prob_train.detach().cpu().float().numpy(),
                "advantages": valid_advantages.detach().cpu().float().numpy(),
                "token_strength": valid_token_strength.detach().cpu().float().numpy(),
                "seq_advantage_sign": valid_seq_advantage.detach()
                .cpu()
                .float()
                .numpy(),
            }

        summaries = {
            "importance_sampling": ir_summary,
            "entropy": entropy_summary,
            "mismatch_kl": mismatch_kl_summary,
            "tv_distance": tv_distance_summary,
            "chi_squared": chi_squared_summary,
            "perplexity_mean": perplexity_mean_summary,
            "prob_train_mean": prob_train_mean_summary,
            "prob_train_min": prob_train_min_summary,
            "prob_train_max": prob_train_max_summary,
            "entropy_pos": entropy_pos_summary,
            "entropy_neg": entropy_neg_summary,
            "perplexity_pos": perplexity_pos_summary,
            "perplexity_neg": perplexity_neg_summary,
            "clipped_fraction": clipped_fraction,
            "clipped_tokens": clipped_tokens,
            "total_kept_tokens": total_kept_tokens,
            "clipped_tokens_pos": clipped_tokens_pos,
            "kept_tokens_pos": kept_tokens_pos,
            "clipped_tokens_neg": clipped_tokens_neg,
            "kept_tokens_neg": kept_tokens_neg,
            "bin_counts": bin_counts,
            "grad_mass": grad_mass_summaries,
            "token_counts": token_count_summaries,
            "plot_data": plot_data,
            "weighted_prob_pos": weighted_prob_pos_summary,
            "weighted_prob_neg": weighted_prob_neg_summary,
        }
        return loss, summaries

    def get_logprobs(
        self,
        model,
        input_ids,
        attention_mask,
        batch_size=None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        batch_size = batch_size or input_ids.size(0)  # chunking for memory peak
        all_logprobs = []
        all_entropies = []
        need_sum_p_sq = self.token_bin_counting
        all_sum_p_sq = [] if need_sum_p_sq else None
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]
            logits_to_keep = attention_mask_batch.size(1) + 1
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
            ).logits
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            targets = input_ids_batch[:, 1:]
            logits = logits[:, -logits_to_keep:]
            logits = logits / self.train_temperature
            logprobs = selective_log_softmax(logits, targets)
            # MODIFIED: use no_grad to save memory
            with torch.no_grad():
                entropies = entropy_from_logits(logits)
            all_logprobs.append(logprobs)
            all_entropies.append(entropies)
            if need_sum_p_sq:
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=-1)  # (B, L-1, V)
                    # Σ p_v² = collision probability (second moment of the distribution)
                    all_sum_p_sq.append(probs.pow(2).sum(-1))  # type: ignore
        logprobs = torch.cat(all_logprobs, dim=0)
        entropies = torch.cat(all_entropies, dim=0)
        sum_p_sq = torch.cat(all_sum_p_sq, dim=0) if all_sum_p_sq is not None else None
        return logprobs, entropies, sum_p_sq

    def update_vllm(self):
        assert self.model is not None
        is_generating = False
        if self.orchestrator:
            is_generating = self.orchestrator.is_generating
        is_generating_list = [is_generating]
        broadcast_object_list(is_generating_list, from_process=0)
        is_generating = is_generating_list[0]

        waits = 0
        while is_generating:
            time.sleep(0.5)
            waits += 1
            if waits % 10 == 0:
                self.logger.info("Waiting for generation to finish before syncing.")
            if self.orchestrator:
                is_generating = self.orchestrator.is_generating
            is_generating_list = [is_generating]
            broadcast_object_list(is_generating_list, from_process=0)
            is_generating = is_generating_list[0]

        # MODIFIED: Log LoRA norm squared before weight sync
        # lora_norm_sq = (
        #     calculate_lora_norm_squared(self.model)
        #     if is_peft_model(self.model)
        #     else 0.0
        # )
        # self.logger.info(f"LoRA parameters L2 norm squared: {lora_norm_sq:.6f}")

        if self.state.global_step >= 0:  # skip first step
            deepspeed_plugin = self.accelerator.state.deepspeed_plugin
            zero_stage_3 = (
                deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
            )
            if zero_stage_3:
                gather_if_zero3 = deepspeed.zero.GatheredParameters
            else:
                gather_if_zero3 = nullcontext
            self.accelerator.wait_for_everyone()
            # MODIFIED: Log current step before weight sync
            self.logger.info(
                f"Starting weight sync to vLLM on step {self.state.global_step}."
            )

            if is_peft_model(self.model):
                # PEFT: gather + merge, then update each parameter
                with gather_if_zero3(list(self.model.parameters())):
                    self.model.merge_adapter()  # type: ignore :(
                    for name, param in self.model.named_parameters():
                        # recover original parameter names
                        name = name.removeprefix("base_model.model.").replace(
                            ".base_layer", ""
                        )
                        if self.model.prefix in name:  # type: ignore :(
                            continue  # discard some parameters
                        if "original_module" in name:  # from modules_to_save
                            continue
                        name = name.replace("modules_to_save.default.", "")
                        if self.client:
                            self.client.update_named_param(name, param.data)
                    self.model.unmerge_adapter()  # type: ignore :(
            else:
                # non-PEFT models: gather + update each parameter individually
                for name, param in self.model.named_parameters():  # type: ignore :(
                    with gather_if_zero3([param]):
                        if self.client:
                            self.client.update_named_param(name, param.data)

            # reset cache + wait for background tasks to complete
            if self.client:
                self.client.reset_prefix_cache()
                while self.client.get_num_background_tasks() > 0:
                    time.sleep(0.5)
                    self.logger.info("Resetting prefix cache.")

        self.accelerator.wait_for_everyone()

    def get_train_dataloader(self):
        class StepsDataset(Dataset):
            def __init__(self, n: int):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return {"labels": 0}

        return DataLoader(StepsDataset(self.max_steps))

    def train(self, **kwargs):
        """Override to pass resume_from_checkpoint to HF Trainer."""
        if self._resume_checkpoint_path:
            kwargs.setdefault("resume_from_checkpoint", self._resume_checkpoint_path)
        return super().train(**kwargs)

    def _inner_training_loop(self, *args, **kwargs):
        """Override to ensure async orchestrator is stopped when training ends"""
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            # cleanup
            if self.orchestrator:
                self.orchestrator.stop()

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs
    ):
        """Run evaluation using the environment's evaluate method through vLLM."""
        self.logger.info(f"Starting evaluation at step {self.state.global_step}")
        eval_start = time.time()
        metrics: dict[str, float] = {}

        if self.accelerator.is_main_process:
            # Wait for any in-flight generation to finish
            if self.orchestrator:
                while self.orchestrator.is_generating:
                    time.sleep(0.5)

            # Create a fresh client for eval
            assert self.vllm_base_url is not None
            eval_client = AsyncOpenAI(
                base_url=self.vllm_base_url,
                api_key="EMPTY",
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=self.args.max_concurrent),
                    timeout=self.args.generation_timeout,
                ),
            )

            eval_inputs = self.env.get_eval_inputs(
                num_examples=self.args.num_eval_examples,
            )
            results = self.env.generate_sync(
                eval_inputs,
                client=eval_client,
                model=self.model_name,
                sampling_args=self.args.sampling_args,
                score_rollouts=True,
                max_concurrent=self.args.max_concurrent,
            )

            # Build metrics
            rewards = np.asarray(results.reward, dtype=np.float32)
            metrics[f"{metric_key_prefix}/reward"] = float(rewards.mean())
            metrics[f"{metric_key_prefix}/reward_std"] = float(rewards.std())

            for rname, rvalues in results.metrics.items():
                if rvalues:
                    metrics[f"{metric_key_prefix}/reward/{rname}"] = float(
                        np.mean(rvalues)
                    )

            # Completion length stats
            comp_lengths = [
                sum(1 for m in c if m.get("role") == "assistant")
                for c in results.completion
            ]
            if comp_lengths:
                metrics[f"{metric_key_prefix}/turns"] = float(np.mean(comp_lengths))

            eval_time = time.time() - eval_start
            metrics[f"{metric_key_prefix}/wall_clock_s"] = eval_time

            # Save eval completions
            import pandas as pd

            eval_dir = os.path.join(self.args.output_dir.replace("checkpoints", "eval"))
            os.makedirs(eval_dir, exist_ok=True)

            def role_content_only(messages):
                if isinstance(messages, str):
                    return messages
                return [
                    {"role": m.get("role", ""), "content": m.get("content", "")}
                    for m in messages
                ]

            table = {
                "step": [str(self.state.global_step)] * len(results.prompt),
                "prompt": [role_content_only(p) for p in results.prompt],
                "completion": [role_content_only(c) for c in results.completion],
                "reward": results.reward,
            }
            for rname, rvalues in results.metrics.items():
                table[rname] = rvalues
            df = pd.DataFrame(table)
            df.to_json(
                os.path.join(eval_dir, f"step_{self.state.global_step}.json"),
                orient="records",
                indent=2,
            )

            self.logger.info(
                f"Evaluation at step {self.state.global_step}: "
                + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )

        # Broadcast metrics to all ranks
        broadcast_list = [metrics]
        broadcast_object_list(broadcast_list, from_process=0)
        metrics = broadcast_list[0]

        # Log metrics (goes to wandb + console)
        for key, value in metrics.items():
            self._metrics["eval"][key].append(value)

        return metrics

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model is not None and self.model.training else "eval"

        if mode == "train":
            fixed_ppl = self.compute_fixed_rollout_perplexity()
            if fixed_ppl is not None:
                self._metrics["train"]["train/fixed_rollout_perplexity"].append(
                    fixed_ppl
                )

        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process:
            print_prompt_completions_sample(
                list(self._textual_logs["prompt"]),  # type: ignore[arg-type]
                list(self._textual_logs["completion"]),  # type: ignore[arg-type]
                list(self._textual_logs["rewards"]["reward"]),  # type: ignore[arg-type]
                self.state.global_step,
            )

            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and wandb.run is not None
            ):
                import pandas as pd

                def role_content_only(messages):
                    if isinstance(messages, str):
                        return messages
                    return [
                        {
                            "role": m.get("role", ""),
                            "content": m.get("content", ""),
                        }
                        for m in messages
                    ]

                prompts_clean = [
                    role_content_only(sanitize_tool_calls(messages_to_printable(p)))
                    for p in self._textual_logs["prompt"]
                ]
                completions_clean = [
                    role_content_only(sanitize_tool_calls(messages_to_printable(c)))
                    for c in self._textual_logs["completion"]
                ]
                table = {
                    "step": [str(self.state.global_step)]
                    * len(self._textual_logs["prompt"]),
                    "prompt": prompts_clean,
                    "completion": completions_clean,
                    **{k: list(v) for k, v in self._textual_logs["rewards"].items()},  # type: ignore[union-attr]
                }
                df = pd.DataFrame(table)
                # MODIFIED: Disable logging textual completions to WandB to save space
                # wandb.log({"completions": wandb.Table(dataframe=df)})
                completions_dir = self.args.output_dir.replace(
                    "checkpoints", "completions"
                )
                os.makedirs(completions_dir, exist_ok=True)
                df.to_json(
                    os.path.join(
                        completions_dir, f"step_{self.state.global_step}.json"
                    ),
                    orient="records",
                    indent=2,
                )

            # clear after logging
            self._textual_logs["prompt"].clear()
            self._textual_logs["completion"].clear()
            for key in self._textual_logs["rewards"]:
                self._textual_logs["rewards"][key].clear()

            # MODIFIED: Log LoRA norm squared after each logging step
            lora_norm_sq = (
                calculate_lora_norm_squared(self.model)
                if is_peft_model(self.model)
                else 0.0
            )
            self.logger.info(
                f"LoRA parameters L2 norm squared after training: {lora_norm_sq:.6f}"
            )

    def log_rollouts(
        self,
        prompts: List[Messages],
        completions: List[Messages],
        rewards_dict: Dict[str, Any],
    ) -> None:
        self._textual_logs["prompt"].extend(prompts)  # type: ignore[union-attr]
        self._textual_logs["completion"].extend(completions)  # type: ignore[union-attr]
        for reward_key in rewards_dict:
            reward_values = rewards_dict[reward_key]
            self._textual_logs["rewards"][reward_key].extend(reward_values)  # type: ignore[union-attr]

    def log_metrics(
        self,
        mode: str,
        batch_metrics: Dict[str, float],
    ) -> None:
        for key, value in batch_metrics.items():
            self._metrics[mode][key].append(value)

    def maybe_clear_cache(self):
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            clear_device_cache()
