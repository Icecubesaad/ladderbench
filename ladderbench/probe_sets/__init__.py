"""Probe set registry.

A probe set is a list of Probe items with exact answers or MCQ letters.
Sets are versioned; results carry the set name + version so runs stay
comparable. Incident probes are static seeds until the sandbox generates
live ones (then a generated set lands here as another module).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    id: str
    tier: str  # easy | medium | hard | incident
    kind: str  # exact | mcq
    question: str
    answer: str


def probe_set_version(name: str) -> str:
    return VERSIONS.get(name, "v0")


VERSIONS = {"core": "v1", "incidents": "v0-static", "hard": "v1",
            "extended": "v1"}

CORE_SIZE_TARGET = 60


def load_probe_set(name: str) -> list[Probe]:
    """Load a probe set by name. Known: core, hard, incidents, extended, all."""
    from .core import PROBES as CORE
    from .hardcore import PROBES as HARD
    from .incidents import PROBES as INCIDENTS

    sets = {"core": CORE, "hard": HARD, "incidents": INCIDENTS}
    if name == "extended":
        return CORE + HARD + INCIDENTS
    if name == "all":
        return CORE + INCIDENTS
    if name not in sets:
        raise SystemExit(
            f"unknown probe set '{name}'. known: core, hard, incidents, "
            f"extended, all")
    return list(sets[name])


def describe() -> str:
    lines = []
    for name in ("core", "hard", "incidents"):
        n = len(load_probe_set(name))
        lines.append(f"  {name:<12} {n:>4} probes  ({VERSIONS[name]})")
    return "\n".join(lines)
