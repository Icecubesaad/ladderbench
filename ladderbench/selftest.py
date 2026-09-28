"""Offline selftest — validates metrics, extraction, the runner loop, and the
sandbox taxonomy with zero network and zero dependencies. Run:

    python -m ladderbench selftest
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from . import probe_sets
from .extraction import check_answer, extract_answer
from .metrics import _rankdata, analyze, spearman
from .runner import DEFAULT_LEVELS, LevelResult, run_level

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def _fake_level(level: str, acc: float, med_tokens: int, empty: float = 0.0) -> LevelResult:
    """Synthetic LevelResult for metric tests."""
    n = 60
    r = LevelResult(level=level)
    r.n = n
    r.n_correct = round(n * acc)
    r.thinking_tokens = [med_tokens] * n
    r.trace_chars = [med_tokens * 4] * n
    r.trace_empty = round(n * empty)
    return r


def test_extraction() -> None:
    print("extraction:")
    check("boxed", extract_answer(r"so \boxed{42} done") == "42")
    check("answer-is", extract_answer("Thus the answer is 1536.") == "1536")
    check("last-number", extract_answer("5! = 120 units total") == "120")
    check("comma-number", extract_answer("total is 1,048,576 bytes") == "1048576")
    check("none-on-empty", extract_answer("") is None)
    check("numeric-eq", check_answer("43", "43"))
    check("float-tol", check_answer("0.5", "0.50"))
    check("mcq-letter", extract_answer("The correct choice is B) OOM kill.", "mcq") == "B")
    check("mcq-wrong", not check_answer("A", "B"))


def test_spearman() -> None:
    print("spearman:")
    ok = lambda x, y: x is not None and abs(x - y) < 1e-9
    check("perfect", ok(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0))
    check("inverse", ok(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0))
    check("ties-ok", abs(spearman([1, 1, 2], [5, 6, 9]) - 0.8660254) < 1e-4)
    check("no-variance", spearman([1, 1, 1], [1, 2, 3]) is None)
    check("avg-ranks", _rankdata([3, 1, 3]) == [2.5, 1.0, 2.5])


def test_verdicts() -> None:
    print("ladder verdicts:")
    ordered = DEFAULT_LEVELS

    healthy = {lv: _fake_level(lv, acc=0.85 - 0.05 * i, med_tokens=4000 - 900 * i)
               for i, lv in enumerate(ordered)}
    a = analyze(healthy, ordered)
    check("healthy", a["ladder_integrity"] == "healthy", str(a["findings"]))

    inverted = {lv: _fake_level(lv, acc=0.8, med_tokens=500 + 900 * i)
                for i, lv in enumerate(ordered)}
    a = analyze(inverted, ordered)
    check("inverted", a["ladder_integrity"] == "degraded"
          and any(f.startswith("inverted") for f in a["findings"]),
          str(a["findings"]))

    collapsed = {lv: _fake_level(lv, acc=0.6, med_tokens=10, empty=0.9)
                 for i, lv in enumerate(ordered)}
    a = analyze(collapsed, ordered)
    check("collapsed", any(f.startswith("collapse") for f in a["findings"]),
          str(a["findings"]))

    flat = {lv: _fake_level(lv, acc=0.8, med_tokens=3000) for lv in ordered}
    a = analyze(flat, ordered)
    check("flattened", any(f.startswith("flattened") for f in a["findings"]),
          str(a["findings"]))

    base = analyze(healthy, ordered)["summary"]
    a = analyze(inverted, ordered, base=base)
    check("calibration-delta", len(a.get("calibration_delta", [])) == len(ordered))


def test_runner_offline() -> None:
    print("runner (offline, fake transport):")
    from .probe_sets.core import PROBES

    def transport(messages, level, temperature=0.0, max_tokens=8192):
        # fake model: answers the first question in every probe correctly
        return {
            "content": "<think>let me work this out carefully step by step "
                       "with a short reasoning trace</think> The answer is 1",
            "reasoning": "let me work this out carefully step by step",
            "completion_tokens": 30,
        }

    res = run_level(None, PROBES[:3], "xhigh", transport=transport)
    check("count", res.n == 3)
    check("tokens-estimated", res.thinking_tokens[0] == 11,
          f"got {res.thinking_tokens[0]}")  # 43-char trace -> ceil(43/4) = 11
    check("empty-not-flagged", res.trace_empty == 0)
    check("usage-captured", res.completion_tokens == [30] * 3)


def test_probe_sets() -> None:
    print("probe sets:")
    core = probe_sets.load_probe_set("core")
    inc = probe_sets.load_probe_set("incidents")
    check("core-60", len(core) == probe_sets.CORE_SIZE_TARGET, f"got {len(core)}")
    check("core-answers-numeric",
          all(p.answer.replace(".", "").replace("-", "").isdigit() for p in core))
    check("ids-unique", len({p.id for p in core + inc}) == len(core) + len(inc))
    check("incidents-mcq", all(p.kind == "mcq" for p in inc))


def test_sandbox() -> None:
    print("sandbox:")
    from .sandbox.episode import EpisodeManifest
    from .sandbox.scenarios import SCENARIOS, render_scenario_yaml

    check("taxonomy-nonempty", len(SCENARIOS) >= 12, f"got {len(SCENARIOS)}")
    chaos = render_scenario_yaml("network-delay", latency="200ms")
    check("chaos-renders", "NetworkChaos" in chaos and "200ms" in chaos)
    patch = render_scenario_yaml("bad-image-tag")
    check("manifest-renders", "image" in patch)
    with tempfile.TemporaryDirectory() as td:
        ep = EpisodeManifest.from_scenario("memory-stress", size="1Gi")
        path = ep.to_json(Path(td) / "ep.json")
        ep2 = EpisodeManifest.from_json(path)
        check("manifest-roundtrip", ep2.fault_class == "OOMKilled"
              and ep2.params == {"size": "1Gi"})
        check("diagnosis-match", ep2.check_diagnosis("oomkilled"))
        check("diagnosis-mismatch", not ep2.check_diagnosis("CPUSaturation"))


def run() -> int:
    print(f"ladderbench selftest\n")
    test_extraction()
    test_spearman()
    test_verdicts()
    test_runner_offline()
    test_probe_sets()
    test_sandbox()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} -> {', '.join(FAILURES)}")
        return 1
    print("ALL PASS")
    return 0
