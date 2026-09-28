"""Verifiable reward oracle for LadderRL (pure python — no torch, no GPU).

Completions are scored against the closed-world incident taxonomy:

* root-cause class match   (exact canonical class, from datafactory)
* recovery-action validity (first key verb of the expected action present)
* format contract          (Root cause: / Recovery: lines present)
* thinking economy         (bounded penalty above a token reference)

Everything here is deterministic given (text, gold, expected) — the property
GRPO needs: a reward that cannot be gamed by style, only by getting it right.
"""

from __future__ import annotations

from dataclasses import dataclass

from ladderbench.datafactory import parse_reply, split_think

_CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class RewardConfig:
    cls_match: float = 1.0        # right fault class
    recovery: float = 0.5         # valid fix action
    format_ok: float = 0.1        # contract followed
    format_bad: float = -0.1      # contract broken
    lambda_think: float = 0.3     # max thinking discount (bounded)
    think_ref_tokens: int = 800   # reference thinking budget


DEFAULT_REWARD = RewardConfig()


def _recovery_valid(recovery: str, expected_action: str) -> bool:
    key = expected_action.split()[0].lower().strip(".,")
    return bool(key) and key in recovery.lower()


def score_text(text: str, gold_class: str, expected_action: str,
               cfg: RewardConfig = DEFAULT_REWARD) -> dict:
    """Score one completion. Returns components + total (deterministic)."""
    parsed = parse_reply(text)
    trace, _ = split_think(text or "")
    think_tokens = int(len(trace) / _CHARS_PER_TOKEN) if trace else 0

    if parsed is None:
        cls_r, rec_r, fmt_r = 0.0, 0.0, cfg.format_bad
    else:
        rc, rec = parsed
        cls_r = cfg.cls_match if gold_class.strip().lower() in rc.strip().lower() else 0.0
        # a "valid" action for a misnamed class is inconsistent guessing —
        # recovery credit is gated on the diagnosis being right
        rec_r = cfg.recovery if cls_r > 0 and _recovery_valid(rec, expected_action) else 0.0
        fmt_r = cfg.format_ok

    think_pen = cfg.lambda_think * min(think_tokens / cfg.think_ref_tokens, 1.0)
    total = cls_r + rec_r + fmt_r - think_pen
    return {
        "total": round(total, 4),
        "cls": cls_r,
        "recovery": rec_r,
        "format": fmt_r,
        "think_penalty": round(think_pen, 4),
        "thinking_tokens": think_tokens,
        "parsed": parsed is not None,
    }


def completion_reward(text: str, instance: dict,
                      cfg: RewardConfig = DEFAULT_REWARD) -> float:
    """Single scalar for TRL's reward_funcs (thin adapter)."""
    return score_text(text, instance["fault_class"],
                      instance["expected_action"], cfg)["total"]
