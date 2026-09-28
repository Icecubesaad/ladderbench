"""LadderRL: effort-aware GRPO harness for Qwen3.8 incident triage.

The training recipe from the architecture doc, in three parts:

* ``reward``    — the verifiable reward oracle (pure python, self-tested)
* ``config``    — GRPO hyperparameters incl. smoke preset
* ``train_grpo``— TRL GRPOTrainer loop (runs Modal-side on the train image)

Reward (per the study design):

    R = class_match * 1.0 + recovery_valid * 0.5 + format * (+/-0.1)
        - lambda_think * min(thinking_tokens / think_ref, 1)

Correctness dominates; the thinking penalty is a bounded discount for wasted
trace — exactly the "answer right, think economically" pressure ThinkingCap
tried to bake in with SFT, done with a verifiable reward instead.
"""

from .reward import score_text, completion_reward

__all__ = ["score_text", "completion_reward"]
