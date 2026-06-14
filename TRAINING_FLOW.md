# Verifiers Training Flow

## What `vf-rl @ config.toml` Does

Creates a **tmux session** with two panes:

```
┌────────────────────────────────────────────────────┐
│  TOP PANE: vLLM Inference Server                   │
│  CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model ... │
├────────────────────────────────────────────────────┤
│  BOTTOM PANE: Trainer                              │
│  CUDA_VISIBLE_DEVICES=1 uv run vf-train @ ...      │
└────────────────────────────────────────────────────┘
```

## Training Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. SETUP                                                           │
│     a) Load model + apply LoRA adapters                             │
│     b) Load environment: vf.load_environment(env_id, **env_args)    │
│     c) Start Orchestrator → connects to vLLM server                 │
│     d) Submit first batch for generation                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. FOR EACH STEP (0 to max_steps):                                 │
│                                                                     │
│     a) update_vllm() - Sync LoRA weights to vLLM server             │
│        - Merge LoRA → base weights → send to vLLM → unmerge         │
│                                                                     │
│     b) Submit next batch for async generation                       │
│                                                                     │
│     c) Get current batch (prompts, completions, rewards, advantages)│
│                                                                     │
│     d) Compute GRPO loss                                            │
│        - importance_ratio = exp(trainer_logp - inference_logp)      │
│        - loss = -importance_ratio * advantages                      │
│                                                                     │
│     e) Backward pass → Update LoRA parameters                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Orchestrator: Async Generation

```
┌─────────────────────────────────────────────────────────────────────┐
│  PER BATCH:                                                         │
│                                                                     │
│  1. Sample examples from dataset                                    │
│                                                                     │
│  2. Generate rollouts via Environment.rollout()                     │
│     - SingleTurnEnv: one response per example                       │
│     - MultiTurnEnv: loop until is_completed() or max_turns          │
│                                                                     │
│  3. Score rollouts with Rubric                                      │
│                                                                     │
│  4. Compute advantages (reward - mean_reward per example)           │
│                                                                     │
│  5. Process → token IDs, logprobs, loss masks                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Multi-Turn Rollout Flow (MultiTurnEnv)

```
┌─────────────────────────────────────────────────────────────────────┐
│  multiturn_env.py:rollout()                                         │
│                                                                     │
│  state = init_state()           # turn = 0                          │
│  state = setup_state(state)                                         │
│                                                                     │
│  while not is_completed():                                          │
│      response = get_model_response()    # Call vLLM                 │
│      state["completion"].append(response)                           │
│      state["turn"] += 1                                             │
│                                                                     │
│      if not is_completed():                                         │
│          env_msgs, state = env_response()  # Environment responds   │
│          state["completion"] += env_msgs                            │
│                                                                     │
│  return completion, state                                           │
└─────────────────────────────────────────────────────────────────────┘
```

## Multi-Agent Specific Flow

```
Planning Agent              MultiAgentEnv                 AnswerAgent
     │                           │                             │
     │  <query>question</query>  │                             │
     │ ────────────────────────► │                             │
     │                           │  extract query              │
     │                           │  answer_agent.answer(query) │
     │                           │ ──────────────────────────► │
     │                           │                             │ retrieve(query)
     │                           │                             │ call LLM
     │                           │   <answer>result</answer>   │
     │                           │ ◄────────────────────────── │
     │ <user_answer>result</user_answer>                       │
     │ ◄──────────────────────── │                             │
     │                           │                             │
     │  <answer>final</answer>   │                             │
     │ ────────────────────────► │  is_completed = True        │
```

## Key Files

| File | Purpose |
|------|---------|
| `verifiers/scripts/rl.py` | `vf-rl` - creates tmux with vLLM + trainer |
| `verifiers/scripts/train.py` | `vf-train` - loads env, creates RLTrainer |
| `verifiers/rl/trainer/trainer.py` | RLTrainer - training loop |
| `verifiers/rl/trainer/config.py` | RLConfig - LoRA args, training args |
| `verifiers/envs/environment.py` | Base Environment, init_state() |
| `verifiers/envs/multiturn_env.py` | MultiTurnEnv, rollout loop, turn tracking |
| `verifiers/envs/multi_agent_env.py` | MultiAgentEnv, env_response() |
