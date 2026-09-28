"""Report building: JSON report + the ladder plot (the hero artifact).

The plot is accuracy vs thinking-tokens with one point per effort level,
joined low-effort -> high-effort, one curve per model in the run (and the
base curve when provided). This is the image that goes at the top of every
README and post — not an architecture diagram.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .runner import DEFAULT_LEVELS, LevelResult


def build_report(
    model: str,
    endpoint: str,
    probe_set: str,
    levels: dict[str, LevelResult],
    analysis: dict,
    ordered: tuple[str, ...] = DEFAULT_LEVELS,
) -> dict:
    return {
        "schema": "ladderbench.report/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ladderbench_version": __version__,
        "model": model,
        "endpoint": endpoint,
        "probe_set": probe_set,
        "levels_ordered": list(ordered),
        "summary": analysis["summary"],
        "ladder_integrity": analysis["ladder_integrity"],
        "findings": analysis["findings"],
        "calibration_delta": analysis.get("calibration_delta"),
        "per_level": {
            lv: {
                "n": res.n,
                "accuracy": round(res.accuracy, 4),
                "median_thinking_tokens": res.median_tokens,
                "empty_trace_rate": round(res.empty_rate, 4),
                "responses": res.responses,
            }
            for lv, res in levels.items()
        },
    }


def save_report(report: dict, out_path: str) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p


def plot_ladder(report: dict, base_report: dict | None, out_path: str) -> str | None:
    """Render the ladder plot. Returns the path, or None when matplotlib is
    unavailable (the JSON report is always complete without it)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ladderbench] matplotlib not installed; skipping ladder plot")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    def _curve(rep: dict, label: str, style: str) -> None:
        s = rep["summary"]
        xs = s["median_thinking_tokens"]
        ys = s["accuracy"]
        ax.plot(xs, ys, style, marker="o", linewidth=2, markersize=9, label=label)
        for lv, x, y in zip(s["levels"], xs, ys):
            ax.annotate(lv, (x, y), textcoords="offset points",
                        xytext=(8, -4), fontsize=9)

    _curve(report, f"{report['model']} (this run)", "o-")
    if base_report:
        _curve(base_report, f"{base_report['model']} (base)", "s--")

    ax.set_xlabel("median thinking tokens (estimated, chars/4)")
    ax.set_ylabel("probe accuracy")
    ax.set_title(
        f"Ladder integrity: {report['model']}\n"
        f"verdict = {report['ladder_integrity']}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)
