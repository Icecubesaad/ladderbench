"""Episode manifests: the reward oracle of the whole project.

An episode = one injected fault + full context (params, expected root cause,
acceptable recovery actions, telemetry pointers). The manifest is written
*before* injection and is ground truth by construction — the agent's job is
to recover it from telemetry alone.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class EpisodeManifest:
    episode_id: str
    scenario: str  # SCENARIOS key
    fault_class: str  # expected root-cause label
    family: str  # chaos | manifest
    params: dict
    expected_actions: list[str]
    injected_at: str
    resolved_at: str | None = None
    telemetry: dict = field(default_factory=dict)  # paths: events/logs/metrics
    notes: str = ""

    @classmethod
    def from_scenario(cls, scenario_name: str, **params) -> "EpisodeManifest":
        from .scenarios import SCENARIOS

        sc = SCENARIOS[scenario_name]
        return cls(
            episode_id=f"ep-{uuid.uuid4().hex[:10]}",
            scenario=sc.name,
            fault_class=sc.fault_class,
            family=sc.family,
            params=dict(params),
            expected_actions=list(sc.expected_actions),
            injected_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    @classmethod
    def from_json(cls, path: str | Path) -> "EpisodeManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def check_diagnosis(self, claimed_class: str) -> bool:
        """Class-level root-cause match (the reward's first term)."""
        return claimed_class.strip().lower() == self.fault_class.lower()
