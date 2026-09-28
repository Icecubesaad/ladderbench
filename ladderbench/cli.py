"""CLI: score / selftest / probe-sets.

    python -m ladderbench score --endpoint http://localhost:8000/v1 \
        --model Qwen/Qwen3.8-27B --probe-set core \
        [--base-curve base.json] [--samples 1] [--out out/report.json]
"""

from __future__ import annotations

import argparse

from . import __version__, probe_sets
from .metrics import analyze
from .report import build_report, plot_ladder, save_report
from .runner import DEFAULT_LEVELS, EndpointClient, LEVELS, run_probe


def _cmd_score(args: argparse.Namespace) -> int:
    problems = probe_sets.load_probe_set(args.probe_set)
    print(
        f"[ladderbench] probe set: {args.probe_set}@{probe_sets.probe_set_version(args.probe_set)}"
        f" ({len(problems)} probes)"
    )
    client = EndpointClient(
        args.endpoint, args.model, api_key=args.api_key,
        kwargs_mode=args.kwargs_mode,
    )
    base_report = None
    if args.base_curve:
        import json
        from pathlib import Path

        base_report = json.loads(
            Path(args.base_curve).read_text(encoding="utf-8")
        )
        print(f"[ladderbench] base curve: {base_report['model']}")

    levels = run_probe(
        client, problems,
        levels=tuple(args.levels.split(",")),
        samples=args.samples,
        max_tokens=args.max_tokens,
    )
    analysis = analyze(levels, DEFAULT_LEVELS,
                       base=base_report["summary"] if base_report else None)
    report = build_report(args.model, args.endpoint, args.probe_set,
                          levels, analysis, DEFAULT_LEVELS)
    out = save_report(report, args.out)
    print(f"[ladderbench] report -> {out}")
    if not args.no_plot:
        plot_path = str(args.out).rsplit(".", 1)[0] + ".png"
        plot_ladder(report, base_report, plot_path)
    print(f"\nverdict: {report['ladder_integrity'].upper()}")
    for f in report["findings"]:
        print(f"  - {f}")
    if report.get("calibration_delta"):
        print("calibration vs base:")
        for row in report["calibration_delta"]:
            print(
                f"  {row['level']:<8} d_acc={row['d_accuracy']:+.3f} "
                f"d_tokens={row['d_thinking_tokens']:+.0f}"
            )
    return 0


def _cmd_selftest(_: argparse.Namespace) -> int:
    from .selftest import run

    return run()


def _cmd_probe_sets(_: argparse.Namespace) -> int:
    print("probe sets:")
    print(probe_sets.describe())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="ladderbench",
        description="Score reasoning-effort ladder integrity of any "
                    "OpenAI-compatible model endpoint.",
    )
    ap.add_argument("--version", action="version",
                    version=f"ladderbench {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("score", help="probe a model at every effort level")
    p.add_argument("--endpoint", required=True,
                   help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--probe-set", default="core",
                   choices=["core", "incidents", "all"])
    p.add_argument("--levels", default=",".join(DEFAULT_LEVELS),
                   help=f"comma-separated, high effort first; known: {LEVELS}")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--base-curve", help="report JSON of the base model for "
                                        "calibration deltas")
    p.add_argument("--api-key")
    p.add_argument("--kwargs-mode", default="auto",
                   choices=["auto", "both", "openai", "llamacpp"],
                   help="how to transmit reasoning_effort to the server")
    p.add_argument("--out", default="out/ladder_report.json")
    p.add_argument("--no-plot", action="store_true")
    p.set_defaults(fn=_cmd_score)

    p = sub.add_parser("selftest", help="run offline self-tests")
    p.set_defaults(fn=_cmd_selftest)

    p = sub.add_parser("probe-sets", help="list available probe sets")
    p.set_defaults(fn=_cmd_probe_sets)

    args = ap.parse_args()
    return args.fn(args)
