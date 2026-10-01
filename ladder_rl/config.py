"""GRPO hyperparameters for LadderRL — presets for smoke and full runs.

Smoke (validation, ~$2-3): tiny prompts/steps — proves reward plumbing,
TRL API surface, and that gradients flow. Full run: your call on budget.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GRPOParams:
    # rollout / group
    num_generations: int = 4        # group size for group-relative advantage
    temperature: float = 0.9
    top_p: float = 0.95
    max_prompt_length: int = 1024    # (unused by current TRL — kept as doc)
    max_completion_length: int = 1024  # HF-rollout pace driver; smoke showed
                                       # 75% clip at 768, 1024 is the balance
                                       # (colocated vLLM rollouts are blocked by
                                       #  TRL's Qwen3.5 param-prefix mismatch)

    # optimization
    learning_rate: float = 1e-5
    lora_r: int = 16
    per_device_batch_size: int = 1
    max_steps: int = 0              # 0 = full epoch over prompts
    total_epochs: int = 1

    # reward
    lambda_think: float = 0.3       # bounded thinking discount
    think_ref_tokens: int = 800

    # dial
    effort: str = "medium"          # xhigh | medium | low (per-run rotation;
                                    # stock TRL sets chat_template_kwargs
                                    # globally, not per-prompt — see README)

    # data
    per_class: int = 40             # 14 scenarios x 40 = 1,960 prompts
    max_prompts: int = 0            # 0 = all

    # rollout engine: False = HF generate (works everywhere),
    # True = colocated vLLM (needs vllm in the image; the production path)
    use_vllm: bool = False

    seed: int = 3407


SMOKE = GRPOParams(
    num_generations=4,
    max_steps=3,
    max_prompts=8,
    max_completion_length=768,
    per_class=1,                    # 14 prompts -> capped by max_prompts
    use_vllm=False,
)


def as_dict(p: GRPOParams) -> dict:
    return {k: getattr(p, k) for k in GRPOParams.__dataclass_fields__}
