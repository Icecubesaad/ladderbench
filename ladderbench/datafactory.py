"""Data factory: textual incident instances for SFT (Phase-0/textual stage).

Each instance is one fault scenario rendered as a realistic incident report:
symptoms + telemetry lines (including the observability hint that makes the
fault class identifiable from text). Ground truth comes from the scenario
taxonomy, so correctness checking is exact. Live tool-calling rollouts in the
kind cluster remain the next stage; this textual simulation is what the two
SFT arms train on, and both arms see the *same prompts* — the only variable
is whether the target contains a thinking trace.

Output contract the model must produce (checked by parse/check):

    ...reasoning (optional <think>...</think>)...
    Root cause: <FaultClass>
    Recovery: <one action>
"""

from __future__ import annotations

import random
import re

from .sandbox.scenarios import SCENARIOS

ROOT_CAUSE_CLASSES: tuple[str, ...] = tuple(
    sorted({sc.fault_class for sc in SCENARIOS.values()})
)

_TAXONOMY = ", ".join(ROOT_CAUSE_CLASSES)

SYSTEM_GEN = (
    "You are an SRE incident responder. Diagnose the root cause from the "
    "report. Think it through step by step, then answer. The root cause MUST "
    f"be exactly one of these classes: {_TAXONOMY}. End with exactly:\n"
    "Root cause: <class>\nRecovery: <action>"
)

_RC = re.compile(r"Root cause:\s*(.+)", re.IGNORECASE)
_REC = re.compile(r"Recovery:\s*(.+)", re.IGNORECASE)
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)

_APPS = ["shop-api", "billing", "inventory", "search", "checkout", "notify"]
_NAMESPACES = ["prod", "staging", "shop", "payments"]


def _symptoms(scenario: str, p: dict, rnd: random.Random) -> tuple[str, str]:
    """(symptom lines, telemetry lines) for one fault instance."""
    t = rnd.choice(["02:14", "03:47", "09:31", "14:08", "18:55"])
    if scenario == "pod-kill":
        sym = (f"Since {t} UTC, pods of app '{p['app']}' restarted unexpectedly; "
               f"service recovered within ~30s each time; no deploys in 7 days.")
        tel = (f"events: Back-off restarting failed container; "
               f"chaos-mesh: podchaos/{p['name']} ACTIVE (action=pod-kill, gracePeriod=0)")
    elif scenario == "container-kill":
        sym = (f"Container '{p['container']}' in app '{p['app']}' was killed at {t} UTC "
               f"and restarted; one-off blip, errors during the gap only.")
        tel = (f"events: container killed; "
               f"chaos-mesh: podchaos/{p['name']} ACTIVE (action=container-kill, container={p['container']})")
    elif scenario == "network-delay":
        sym = (f"p99 latency {p['lat']} (baseline 45ms) on app '{p['app']}'; "
               f"CPU and memory normal; no code changes in 7 days.")
        tel = f"chaos-mesh: networkchaos/{p['name']} ACTIVE (action=delay, latency={p['lat']})"
    elif scenario == "network-loss":
        sym = (f"app '{p['app']}' drops ~{p['loss']}% of requests intermittently; "
               f"retries elevated; upstream healthy.")
        tel = f"chaos-mesh: networkchaos/{p['name']} ACTIVE (action=loss, loss={p['loss']}%)"
    elif scenario == "dns-failure":
        sym = (f"app '{p['app']}' intermittently fails to resolve service names; "
               f"connections via direct IPs still work.")
        tel = f"chaos-mesh: networkchaos/{p['name']} ACTIVE (action=netem, dns disruption)"
    elif scenario == "cpu-stress":
        sym = (f"app '{p['app']}' pegged at 100% CPU with throttling since {t} UTC; "
               f"memory normal; traffic unchanged.")
        tel = f"chaos-mesh: stresschaos/{p['name']} ACTIVE (stressors.cpu load={p['load']})"
    elif scenario == "memory-stress":
        sym = (f"container in app '{p['app']}' climbed to its memory limit and was "
               f"killed: Exit Code 137, Reason OOMKilled; restarting repeatedly.")
        tel = f"chaos-mesh: stresschaos/{p['name']} ACTIVE (stressors.memory size={p['size']})"
    elif scenario == "io-latency":
        sym = (f"disk operations on app '{p['app']}' slowed massively; DB queries queue; "
               f"CPU idle; started {t} UTC.")
        tel = (f"chaos-mesh: iochaos/{p['name']} ACTIVE (action=latency, "
               f"volumePath={p['volume_path']}, delay={p['delay']})")
    elif scenario == "bad-image-tag":
        sym = f"rollout of app '{p['app']}' is stuck; no new pods become Ready."
        tel = (f"events: Failed to pull image 'registry.local/{p['app']}:{p['tag']}': "
               f"manifest unknown; Back-off pulling image")
    elif scenario == "bad-command":
        sym = (f"pods of app '{p['app']}' crash-loop since the last deploy; "
               f"container logs are empty; process exits immediately after start.")
        tel = "describe: State Waiting Reason CrashLoopBackOff, Last State Exit Code 0"
    elif scenario == "missing-env":
        sym = (f"pods of app '{p['app']}' crash on boot since deploy r{p['rev']} "
               f"removed an environment variable; logs show a required-key error.")
        tel = (f"logs: KeyError: '{p['env_key']}' — required environment variable "
               f"not set; describe: CrashLoopBackOff")
    elif scenario == "probe-mismatch":
        sym = (f"pods of app '{p['app']}' restart every ~40s despite healthy app logs "
               f"serving traffic fine between restarts.")
        tel = (f"events: Liveness probe failed: HTTP 404 on path '{p['path']}'; "
               f"Killing container")
    elif scenario == "quota-exceeded":
        sym = f"new pods of app '{p['app']}' never start; scheduler rejects them."
        tel = (f"events: exceeded quota: compute-resources, requested: "
               f"limits.memory={p['mem']}; namespace quota nearly exhausted")
    elif scenario == "pvc-mismatch":
        sym = f"pods of app '{p['app']}' stuck in ContainerCreating; never Running."
        tel = (f"events: MountVolume.SetUp failed for volume 'data': references "
               f"non-existent PVC '{p['claim']}'")
    else:
        raise ValueError(scenario)
    return sym, tel


_PARAMS = {
    "pod-kill": lambda r: {"name": f"ladder-pk-{r.randint(100,999)}"},
    "container-kill": lambda r: {"name": f"ladder-ck-{r.randint(100,999)}",
                                 "container": r.choice(["app", "sidecar", "worker"])},
    "network-delay": lambda r: {"name": f"ladder-nd-{r.randint(100,999)}",
                                "lat": r.choice(["200ms", "350ms", "500ms"])},
    "network-loss": lambda r: {"name": f"ladder-nl-{r.randint(100,999)}",
                               "loss": r.choice([15, 30, 45])},
    "dns-failure": lambda r: {"name": f"ladder-dns-{r.randint(100,999)}"},
    "cpu-stress": lambda r: {"name": f"ladder-cpu-{r.randint(100,999)}",
                             "load": r.choice([80, 95, 100])},
    "memory-stress": lambda r: {"name": f"ladder-mem-{r.randint(100,999)}",
                                "size": r.choice(["1Gi", "2Gi", "4Gi"])},
    "io-latency": lambda r: {"name": f"ladder-io-{r.randint(100,999)}",
                             "volume_path": "/data", "delay": r.choice(["50ms", "120ms"])},
    "bad-image-tag": lambda r: {"tag": f"v{r.randint(90,99)}.{r.randint(0,9)}"},
    "bad-command": lambda r: {},
    "missing-env": lambda r: {"rev": r.randint(40, 80),
                              "env_key": r.choice(["DATABASE_URL", "REDIS_HOST", "API_KEY"])},
    "probe-mismatch": lambda r: {"path": r.choice(["/healthz-wrong", "/health", "/livez"])},
    "quota-exceeded": lambda r: {"mem": r.choice(["2Gi", "4Gi"])},
    "pvc-mismatch": lambda r: {"claim": r.choice(["cache-claim", "data-claim", "logs-claim"])},
}


def build_instances(seed: int = 3407, per_class: int = 4) -> list[dict]:
    """Deterministic incident instances across the whole fault taxonomy."""
    rnd = random.Random(seed)
    out = []
    for i, (name, sc) in enumerate(sorted(SCENARIOS.items())):
        for j in range(per_class):
            p = _PARAMS[name](rnd)
            p.setdefault("app", rnd.choice(_APPS))
            ns = rnd.choice(_NAMESPACES)
            sym, tel = _symptoms(name, p, rnd)
            question = (
                f"Incident report — namespace '{ns}', app '{p['app']}'.\n"
                f"Symptoms:\n{sym}\nTelemetry:\n{tel}\n\n"
                "Diagnose the root cause and give ONE recovery action. End with "
                "exactly:\nRoot cause: <class>\nRecovery: <action>"
            )
            out.append({
                "id": f"{name}-{j:02d}",
                "scenario": name,
                "fault_class": sc.fault_class,
                "expected_action": sc.expected_actions[0],
                "question": question,
            })
    return out


def parse_reply(text: str) -> tuple[str, str] | None:
    """Extract (root_cause, recovery); None when the contract is violated."""
    if not text:
        return None
    rc = _RC.findall(text)
    rec = _REC.findall(text)
    if not rc or not rec:
        return None
    return rc[-1].strip(), rec[-1].strip()


def check(reply: tuple[str, str], instance: dict) -> bool:
    """Class-level match: the canonical fault class must appear in the model's
    Root cause line (models write verbose lines like 'CrashLoopBackOff due to
    a bad entrypoint' — containment accepts those, different mechanisms fail)."""
    rc, rec = reply
    return (instance["fault_class"].strip().lower() in rc.strip().lower()
            and bool(rec.strip()))


def split_think(text: str) -> tuple[str, str]:
    """(trace, final). trace = <think> content when tags exist, else the
    model's plain prose before the 'Root cause:' line (visible CoT)."""
    if "</think>" in text:
        trace, final = text.rsplit("</think>", 1)
        trace = trace.split("<think>")[-1]
        return trace.strip(), final.strip()
    final = text.strip()
    idx = final.find("Root cause:")
    if idx > 0:
        return final[:idx].strip(), final[idx:].strip()
    return "", final


def direct_answer(instance: dict) -> str:
    return f"Root cause: {instance['fault_class']}\nRecovery: {instance['expected_action']}"


def reasoning_answer(trace: str, instance: dict) -> str:
    return f"<think>\n{trace.strip()}\n</think>\n\n{direct_answer(instance)}"


def render_training_text(tokenizer, system: str, user: str, assistant: str) -> str:
    """Render one conversation with the model's own chat template, then
    normalize think tags: templates that auto-emit an empty <think></think>
    before assistant content would double-wrap — collapse that to one block."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if "<think>" in assistant:
        for empty in ("<think>\n\n</think>\n\n", "<think>\n\n</think>",
                      "<think></think>"):
            text = text.replace(empty, "", 1)
    return text
