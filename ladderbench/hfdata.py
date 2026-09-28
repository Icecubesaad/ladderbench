"""External HuggingFace dataset adapters -> the LADDER instance schema.

Bridges public datasets into the exact format the training/eval pipeline
consumes (same dict schema as datafactory.build_instances: id, fault_class,
question, expected_action), so external data flows through rejection
sampling -> SFT -> LadderBench unchanged.

Adapters:
* ``devops-incident-response`` — 10 real-world-style DevOps incidents (HF:
  Snaseem2026/devops-incident-response). Free-text root causes are mapped
  onto the 13-class taxonomy by keyword rules; unmappable rows are reported,
  never silently forced.

Design rule: an adapter may only emit instances whose fault_class is in
datafactory's ROOT_CAUSE_CLASSES, and every adapter reports its coverage so
mapping quality is visible, not assumed.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from .datafactory import ROOT_CAUSE_CLASSES
from .sandbox.scenarios import SCENARIOS

# canonical class -> detection patterns (checked against title + root_cause +
# symptoms, case-insensitive). Ordered; first hit wins for multi-class text.
TAXONOMY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "OOMKilled": (r"oomkill", r"out of memory", r"exit code 137", r"oom\b"),
    "ImagePullBackOff": (r"imagepull", r"pull image", r"manifest unknown",
                         r"image not found", r"errimagepull"),
    "CrashLoopBackOff": (r"crashloop", r"crash looping", r"restart loop",
                         r"exits? immediately", r"exit code 0\b"),
    "LivenessProbeFailure": (r"liveness", r"readiness", r"health check fail",
                             r"probe fail"),
    "PVCMountFailure": (r"\bpvc\b", r"persistent volume", r"volume mount",
                        r"failedmount"),
    "DNSResolutionFailure": (r"\bdns\b", r"name resolution", r"nslookup",
                             r"service discovery"),
    "ResourceQuotaExhausted": (r"quota", r"resource limit exceed"),
    "NetworkLatencyInjected": (r"latency spike", r"high latency"),
    "PacketLossInjected": (r"packet loss", r"network partition"),
    "CPUSaturation": (r"cpu (saturation|throttl|pegged)", r"high cpu"),
    "PodKilled": (r"pod (was )?kill", r"evicted"),
    "ContainerKilled": (r"container (was )?kill", r"container terminated"),
    "StorageLatencyInjected": (r"disk (i/o )?latency", r"storage slow"),
}

_CLASS_RE = {c: re.compile("|".join(pats), re.IGNORECASE)
             for c, pats in TAXONOMY_KEYWORDS.items()}


def classify(*texts: str) -> str | None:
    """Map free-text incident fields onto the taxonomy; None = unmappable.

    Priority pass: the root_cause field (usually texts[0]) is checked for
    every class before falling back to title/symptoms — titles describe
    symptoms ('pods crashlooping') while root_cause states the mechanism
    ('OOMKilled'), and the mechanism is the ground truth.
    """
    primary, secondary = (texts[0] or ""), " ".join(t for t in texts[1:] if t)
    for cls in ROOT_CAUSE_CLASSES:
        if _CLASS_RE[cls].search(primary):
            return cls
    for cls in ROOT_CAUSE_CLASSES:
        if _CLASS_RE[cls].search(secondary):
            return cls
    return None


def default_action(fault_class: str) -> str:
    for sc in SCENARIOS.values():
        if sc.fault_class == fault_class:
            return sc.expected_actions[0]
    return "apply the standard remediation for this fault class"


def _parse_listish(value) -> list[str]:
    """Fields arrive as lists or python-literal strings of lists."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        return [value]
    return []


def to_instance(row: dict, idx: int) -> dict | None:
    """One dataset row -> one pipeline instance (or None if unmappable)."""
    root = str(row.get("root_cause", ""))
    title = str(row.get("title", ""))
    fault_class = classify(root, title, str(row.get("symptoms", "")))
    if fault_class is None:
        return None
    symptoms = _parse_listish(row.get("symptoms"))
    steps = _parse_listish(row.get("resolution_steps"))
    action = next((s for s in steps
                   if re.match(r"^\s*(roll ?back|restart|delete|raise|correct|"
                               r"restore|fix|create|set|increase|scale|update|"
                               r"add|remove|replace|clear|purge|recycle)\b",
                               s, re.IGNORECASE)), "")
    question = (
        f"Incident report — {row.get('severity', 'unknown')} severity, "
        f"category '{row.get('category', 'unknown')}'.\n"
        f"Title: {title}\nDescription: {row.get('description', '')}\n"
        + (f"Symptoms:\n" + "\n".join(f"- {s}" for s in symptoms) + "\n"
           if symptoms else "")
        + "\nDiagnose the root cause and give ONE recovery action. End with "
        "exactly:\nRoot cause: <class>\nRecovery: <action>"
    )
    return {
        "id": f"hf-{row.get('incident_id', idx)}",
        "scenario": f"hf:{row.get('incident_id', idx)}",
        "fault_class": fault_class,
        "expected_action": action or default_action(fault_class),
        "question": question,
    }


DATASETS_SERVER = ("https://datasets-server.huggingface.co/rows"
                   "?dataset={dataset}&config={config}&split={split}"
                   "&offset={offset}&length=100")


def load_devops_incident_response(max_rows: int = 100) -> list[dict]:
    """Fetch + map the HF dataset. Returns (instances, report)."""
    instances, unmapped, rows_seen, offset = [], [], 0, 0
    while rows_seen < max_rows:
        url = DATASETS_SERVER.format(dataset="Snaseem2026/devops-incident-response",
                                     config="default", split="train", offset=offset)
        with urllib.request.urlopen(url, timeout=30) as r:
            page = json.load(r)
        rows = page.get("rows", [])
        if not rows:
            break
        for i, entry in enumerate(rows):
            row = entry["row"]
            rows_seen += 1
            inst = to_instance(row, offset + i)
            if inst is None:
                unmapped.append({"id": row.get("incident_id"),
                                 "root_cause": str(row.get("root_cause", ""))[:120]})
            else:
                instances.append(inst)
        offset += len(rows)
        if rows_seen >= page.get("num_rows_total", rows_seen):
            break
    coverage: dict[str, int] = {}
    for inst in instances:
        coverage[inst["fault_class"]] = coverage.get(inst["fault_class"], 0) + 1
    report = {"rows_seen": rows_seen, "mapped": len(instances),
              "unmapped": len(unmapped), "coverage": coverage,
              "unmapped_details": unmapped}
    return instances, report


SFT100K = "stindardlogic/devops-kubernetes-sft-100k"

_GROUNDING_CATEGORIES = (
    "kubernetes_troubleshooting", "kubernetes_debugging",
    "monitoring_observability", "kubernetes_security",
)


def _fetch_json(url: str, tries: int = 6) -> dict:
    """GET with backoff — datasets-server rate-limits anonymous cloud IPs."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={"User-Agent": "ladder-vet"}), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < tries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def load_sft100k_grounding(n: int = 5000, max_pages: int = 60) -> tuple[list[dict], dict]:
    """Layer-B grounding rows from the 100k Apache-2.0 DevOps/K8s SFT set.

    Stratified: troubleshooting/debugging/observability/security categories
    are collected first (that is the incident agent's neighborhood), then
    uniform fill. Each row becomes a single-turn user/assistant message pair
    (the dataset's answers carry no reasoning traces — this layer teaches
    domain fluency, not thinking style, and is a constant across study arms).
    Returns (rows, stats): rows = [{'user':..., 'assistant':...}].
    """
    rows: list[dict] = []
    stats = {"seen": 0, "kept": 0, "skipped_len": 0, "pages": 0}
    for offset in range(0, max_pages * 100, 100):
        url = (f"https://datasets-server.huggingface.co/rows?dataset="
               f"{SFT100K.replace('/', '%2F')}&config=default&split=train"
               f"&offset={offset}&length=100")
        page = _fetch_json(url)
        time.sleep(1.5)  # stay under the anonymous rate limit
        entries = page.get("rows", [])
        stats["pages"] += 1
        if not entries:
            break
        for e in entries:
            stats["seen"] += 1
            conv = e["row"].get("conversations") or []
            human = next((c["value"] for c in conv if c.get("from") == "human"), None)
            gpt = next((c["value"] for c in conv if c.get("from") == "gpt"), None)
            if not human or not gpt:
                continue
            if not (100 <= len(gpt) <= 6000):  # drop stubs and mega-answers
                stats["skipped_len"] += 1
                continue
            category = (e["row"].get("metadata") or {}).get("category", "")
            rows.append({"user": human, "assistant": gpt,
                         "priority": category in _GROUNDING_CATEGORIES})
            stats["kept"] += 1
        if len(rows) >= n * 2:  # 2x oversample before stratified cut
            break
    prioritized = [r for r in rows if r["priority"]][:n]
    rest = [r for r in rows if not r["priority"]]
    picked = prioritized + rest[: max(0, n - len(prioritized))]
    stats["prioritized_kept"] = len(prioritized)
    return picked[:n], stats


def load_rcaeval_cases(suite: str = "re1", max_cases: int = 50) -> list[dict]:
    """RCAEval real-telemetry cases (HF: phamquiluan/RCAEval, parquet).

    Runs where pyarrow/pandas exist (Modal image). Returns raw case dicts;
    mapping into the taxonomy goes through the same classify() rules plus a
    per-suite fault-type table — kept separate because RCAEval's labels are
    service-level and need the distiller pass before SFT use.
    """
    import pandas as pd  # lazy: Modal image dependency

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="phamquiluan/RCAEval", repo_type="dataset",
                           filename=f"{suite}/cases.parquet")
    df = pd.read_parquet(path).head(max_cases)
    return df.to_dict("records")
