"""Ladder-integrity metrics: monotonicity, calibration, collapse.

All statistics are dependency-free (Spearman via average ranks). Levels are
ordered high effort -> low; ``effort_rank`` is reversed so that a *positive*
Spearman against effort means "more effort, more X" (healthy for both tokens
and accuracy).
"""

from __future__ import annotations

from .runner import LevelResult

# thresholds chosen to be decisive on the small probe sets this tool ships with
INVERTED_R = -0.60
FLAT_R = 0.30
MISALIGNED_R = -0.50
COLLAPSE_EMPTY_RATE = 0.50
COLLAPSE_TOKEN_FLOOR = 20.0


def _rankdata(xs: list[float]) -> list[float]:
    """Average ranks, 1-based, ties get the mean rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rho, or None when either side has no variance (undefined)."""
    if len(a) != len(b) or len(a) < 2:
        return None
    ra, rb = _rankdata(a), _rankdata(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def summarize(levels: dict[str, LevelResult], ordered: tuple[str, ...]) -> dict:
    """Aggregate per-level results into the ladder metrics block."""
    present = [lv for lv in ordered if lv in levels and levels[lv].n > 0]
    k = len(present)
    effort = list(range(k - 1, -1, -1))  # first level = most effort = highest rank
    tokens = [levels[lv].median_tokens for lv in present]
    acc = [levels[lv].accuracy for lv in present]
    r_tokens = spearman(effort, tokens)
    r_acc = spearman(effort, acc)
    top = levels[present[0]]
    token_spread = (max(tokens) / min(tokens)) if min(tokens) > 0 else float("inf")
    return {
        "levels": present,
        "accuracy": acc,
        "median_thinking_tokens": tokens,
        "spearman_tokens_vs_effort": r_tokens,
        "spearman_accuracy_vs_effort": r_acc,
        "top_level": {
            "level": present[0],
            "accuracy": top.accuracy,
            "median_thinking_tokens": top.median_tokens,
            "empty_trace_rate": top.empty_rate,
        },
        "token_spread_max_over_min": token_spread,
    }


def _verdict(s: dict) -> tuple[str, list[str]]:
    findings: list[str] = []
    r_tok = s["spearman_tokens_vs_effort"]
    r_acc = s["spearman_accuracy_vs_effort"]
    top_empty = s["top_level"]["empty_trace_rate"]
    top_tokens = s["top_level"]["median_thinking_tokens"]

    if top_empty >= COLLAPSE_EMPTY_RATE and top_tokens < COLLAPSE_TOKEN_FLOOR:
        findings.append(
            f"collapse: {top_empty:.0%} empty thinking traces at "
            f"{s['top_level']['level']} (median {top_tokens:.0f} tokens)"
        )
    if r_tok is not None and r_tok <= INVERTED_R:
        findings.append(f"inverted dial: tokens vs effort rho={r_tok:.2f}")
    spread = s["token_spread_max_over_min"]
    # a zero-variance token curve makes Spearman undefined — that IS flat
    flat_r = r_tok if r_tok is not None else 0.0
    if abs(flat_r) < FLAT_R and spread < 1.5:
        findings.append(
            f"flattened dial: tokens vs effort rho={r_tok if r_tok is not None else float('nan'):.2f}, "
            f"spread={spread:.2f}x"
        )
    if r_acc is not None and r_acc <= MISALIGNED_R:
        findings.append(f"misaligned: accuracy vs effort rho={r_acc:.2f}")

    if not findings:
        return ("healthy", [])
    return ("degraded", findings)


def analyze(levels: dict[str, LevelResult], ordered: tuple[str, ...],
            base: dict | None = None) -> dict:
    """Full analysis block: metrics, verdict, findings, optional calibration
    delta against a reference (base-model) report's ``summary`` section."""
    s = summarize(levels, ordered)
    verdict, findings = _verdict(s)
    out = {
        "summary": s,
        "ladder_integrity": verdict,
        "findings": findings,
    }
    if base:
        out["calibration_delta"] = _calibration(s, base)
    return out


def _calibration(s: dict, base: dict) -> list[dict]:
    """Per-level deltas of (accuracy, median tokens) vs the base curve."""
    blv = base.get("levels", [])
    bacc = base.get("accuracy", [])
    btok = base.get("median_thinking_tokens", [])
    rows = []
    for i, lv in enumerate(s["levels"]):
        if lv in blv:
            j = blv.index(lv)
            rows.append(
                {
                    "level": lv,
                    "d_accuracy": round(s["accuracy"][i] - bacc[j], 4),
                    "d_thinking_tokens": round(
                        s["median_thinking_tokens"][i] - btok[j], 1
                    ),
                }
            )
    return rows
