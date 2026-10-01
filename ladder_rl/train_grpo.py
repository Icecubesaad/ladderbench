"""GRPO training loop for LadderRL (runs Modal-side; see modal_ladder.run_grpo).

Stock TRL GRPOTrainer, verifiable reward from ladder_rl.reward, prompts built
from the same incident generator as the SFT arms (closed-world taxonomy, so
the reward is exact). Effort dial is applied globally per run via
chat_template_kwargs — rotate effort across runs until per-prompt sampling
exists in TRL (documented limitation, config.EFFORT note).
"""

from __future__ import annotations

from typing import Any

from .config import GRPOParams, SMOKE, as_dict


def _decode_completion(c: Any, tokenizer) -> str:
    """TRL versions hand completions as str | list[int] | tensor | chunked
    lists — normalize all of them to text."""
    if isinstance(c, str):
        return c
    if hasattr(c, "tolist"):
        c = c.tolist()
    if isinstance(c, list):
        if c and isinstance(c[0], int):
            return tokenizer.decode(c)
        flat: list[int] = []
        for chunk in c:
            if hasattr(chunk, "tolist"):
                chunk = chunk.tolist()
            if isinstance(chunk, int):
                flat.append(chunk)
            elif isinstance(chunk, list):
                flat.extend(chunk)
        if flat:
            return tokenizer.decode(flat)
    return str(c)


def train(cache: str = "/cache", model_id: str = "", smoke: bool = False,
          overrides: dict | None = None) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from ladderbench.datafactory import SYSTEM_GEN, build_instances
    from ladder_rl.reward import RewardConfig, completion_reward

    p = SMOKE if smoke else GRPOParams()
    if overrides:
        from dataclasses import replace
        p = replace(p, **{k: v for k, v in overrides.items()
                          if v is not None and hasattr(p, k)})

    # ---- prompts (gold labels ride along as dataset columns -> reward kwargs)
    instances = build_instances(seed=p.seed, per_class=p.per_class)
    if p.max_prompts:
        instances = instances[: p.max_prompts]
    rows = [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_GEN},
                {"role": "user", "content": inst["question"]},
            ],
            "fault_class": inst["fault_class"],
            "expected_action": inst["expected_action"],
        }
        for inst in instances
    ]
    ds = Dataset.from_list(rows)
    print(f"[ladderRL] prompts={len(ds)} params={as_dict(p)}")

    # ---- model + LoRA
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    lora = LoraConfig(
        r=p.lora_r, lora_alpha=p.lora_r, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # ---- verifiable reward bridge
    rcfg = RewardConfig(lambda_think=p.lambda_think,
                        think_ref_tokens=p.think_ref_tokens)

    def reward_fn(completions, fault_class=None, expected_action=None,
                  **_: Any) -> list[float]:
        out = []
        for i, c in enumerate(completions):
            text = _decode_completion(c, tokenizer)
            fc = fault_class[i] if isinstance(fault_class, list) else fault_class
            ea = expected_action[i] if isinstance(expected_action, list) else expected_action
            out.append(completion_reward(
                text, {"fault_class": fc, "expected_action": ea}, rcfg))
        return out

    # ---- GRPO config
    args = GRPOConfig(
        output_dir=f"{cache}/tmp/grpo{'_smoke' if smoke else ''}",
        per_device_train_batch_size=p.per_device_batch_size,
        num_generations=p.num_generations,
        generation_batch_size=p.num_generations,  # must be divisible by group size
        # max_prompt_length was removed from GRPOConfig in current TRL —
        # incident prompts are <500 tokens by construction, no truncation needed
        max_completion_length=p.max_completion_length,
        generation_kwargs={"temperature": p.temperature, "top_p": p.top_p},
        learning_rate=p.learning_rate,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_steps=p.max_steps or -1,
        num_train_epochs=p.total_epochs,
        chat_template_kwargs={"reasoning_effort": p.effort},
        beta=0.0,  # no KL-vs-ref: keeps the ref-model copy out of VRAM
        use_vllm=p.use_vllm,
        vllm_gpu_memory_utilization=0.4,  # colocated rollout engine budget
        vllm_max_model_length=4096,  # prompts ~500 + completion 1536; never
                                     # size KV for Qwen's 262K context here
        seed=p.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()

    # ---- save merged checkpoint (smoke saves too — proves the export path)
    out_name = "grpo_smoke" if smoke else "grpo"
    merged = trainer.model.merge_and_unload()
    out_dir = f"{cache}/checkpoints/{out_name}"
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # ---- reward trajectory (first vs last logged step)
    reward_hist = [s.get("reward") for s in trainer.state.log_history
                   if "reward" in s]
    return {
        "run": out_name,
        "prompts": len(ds),
        "steps": args.max_steps if args.max_steps > 0 else
                 len(ds) * p.total_epochs,
        "reward_first": reward_hist[0] if reward_hist else None,
        "reward_last": reward_hist[-1] if reward_hist else None,
        "reward_history": reward_hist[:50],
        "saved": out_dir,
        "params": as_dict(p),
    }
