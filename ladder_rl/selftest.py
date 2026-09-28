"""Offline selftest for the LadderRL reward oracle. Run:

    python -m ladder_rl.selftest
"""

from __future__ import annotations

from .reward import DEFAULT_REWARD, RewardConfig, completion_reward, score_text

FAILURES: list[str] = []
GOLD = {"fault_class": "OOMKilled", "expected_action": "raise the memory limit"}

CORRECT = "Root cause: OOMKilled\nRecovery: raise the memory limit"
PREAMBLE = (
    "The container exceeded its memory limit; exit code 137 confirms.\n"
    "Root cause: OOMKilled\nRecovery: raise the memory limit"
)
LONG_TRACE = "thoughtful trace" * 1200  # ~19k chars
# visible-CoT style (no tags) -> split_think fallback counts pre-answer prose
CORRECT_LONG_THINK = (
    LONG_TRACE + " relevant reasoning.\n"
    "Root cause: OOMKilled\nRecovery: raise the memory limit"
)
WRONG_CLASS = "Root cause: CrashLoopBackOff\nRecovery: raise the memory limit"
WRONG_ACTION = "Root cause: OOMKilled\nRecovery: restart the deployment"
BROKEN_FORMAT = "It's obviously an OOM problem, raise the limit."
EMPTY = ""


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def main() -> int:
    cfg = DEFAULT_REWARD
    print("reward oracle:")

    r = score_text(CORRECT, GOLD["fault_class"], GOLD["expected_action"])
    check("correct+exact contract = full credit",
          abs(r["total"] - 1.6) < 1e-9, str(r))

    r = score_text(PREAMBLE, GOLD["fault_class"], GOLD["expected_action"])
    check("correct+small preamble: tiny bounded penalty",
          0 < r["think_penalty"] < 0.05 and r["total"] > 1.55, str(r))

    r = score_text(CORRECT_LONG_THINK, GOLD["fault_class"],
                   GOLD["expected_action"])
    check("long think: penalty capped at lambda",
          abs(r["think_penalty"] - cfg.lambda_think) < 1e-9, str(r))
    check("long think total", abs(r["total"] - (1.6 - cfg.lambda_think)) < 1e-9,
          str(r))
    check("long think tokens counted", r["thinking_tokens"] > 1000, str(r))

    r = score_text(WRONG_CLASS, GOLD["fault_class"], GOLD["expected_action"])
    check("wrong class: cls=0, recovery gated off", r["cls"] == 0.0
          and r["recovery"] == 0.0, str(r))
    check("wrong class bounded", -0.4 < r["total"] <= 0.15, str(r))

    r = score_text(WRONG_ACTION, GOLD["fault_class"], GOLD["expected_action"])
    check("right class + wrong action: partial credit",
          r["cls"] == 1.0 and r["recovery"] == 0.0, str(r))

    r = score_text(BROKEN_FORMAT, GOLD["fault_class"], GOLD["expected_action"])
    check("broken format: negative", r["total"] < 0 and r["parsed"] is False,
          str(r))

    r = score_text(EMPTY, GOLD["fault_class"], GOLD["expected_action"])
    check("empty text worst", r["total"] <= 0.0, str(r))

    hi = completion_reward(CORRECT, GOLD)
    lo = completion_reward(EMPTY, GOLD)
    check("bounds", 1.6 >= hi and lo >= -0.4, f"hi={hi} lo={lo}")
    check("deterministic", hi == completion_reward(CORRECT, GOLD))

    r = score_text(CORRECT, GOLD["fault_class"], GOLD["expected_action"],
                   RewardConfig(lambda_think=1.0, think_ref_tokens=100))
    check("custom cfg respected", abs(r["think_penalty"]) < 1e-9,
          "exact contract has no think -> 0 even with harsh cfg: " + str(r))

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} -> {', '.join(FAILURES)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
