"""Fault taxonomy: every scenario we can inject, and how to render it.

Two families, mirroring how real incidents actually happen:

* ``chaos``   — Chaos Mesh custom resources (network, stress, io, dns...),
                parameterized YAML rendered with pyyaml.
* ``manifest`` — Kubernetes-native faults via wrong manifests (bad image tag,
                bad command, missing env, probe mismatch, quota, PVC mismatch).

Each scenario carries its expected root-cause class and acceptable recovery
actions — that pair is the GRPO reward oracle later. One scenario template x
parameter variation = many distinct episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str  # chaos | manifest
    fault_class: str  # the label the agent must produce
    description: str
    expected_actions: tuple[str, ...]
    template: dict = field(default_factory=dict)  # chaos CR template
    manifest_patch: dict = field(default_factory=dict)  # manifest fault spec

    @property
    def kind(self) -> str:
        return "chaos-mesh" if self.family == "chaos" else "k8s-manifest"


SCENARIOS: dict[str, Scenario] = {
    # ---------- chaos-mesh family ----------
    "pod-kill": Scenario(
        "pod-kill", "chaos", "PodKilled",
        "Random pod deletion on a target app (recovers via replication).",
        ("restart deployment", "verify replicas recover"),
        template={
            "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "PodChaos",
            "metadata": {"name": "{name}", "namespace": "chaos-testing"},
            "spec": {
                "action": "pod-kill", "mode": "fixed", "value": "1",
                "gracePeriod": 0,
                "selector": {"namespaces": ["{namespace}"],
                             "labelSelectors": {"app": "{app}"}},
                "duration": "{duration}",
            },
        },
    ),
    "container-kill": Scenario(
        "container-kill", "chaos", "ContainerKilled",
        "Kill a single container inside a pod (app-level transient failure).",
        ("restart deployment", "verify service recovers"),
        template={
            "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "PodChaos",
            "metadata": {"name": "{name}", "namespace": "chaos-testing"},
            "spec": {
                "action": "container-kill", "mode": "one",
                "containerNames": ["{container}"],
                "selector": {"namespaces": ["{namespace}"],
                             "labelSelectors": {"app": "{app}"}},
                "duration": "{duration}",
            },
        },
    ),
    "network-delay": Scenario(
        "network-delay", "chaos", "NetworkLatencyInjected",
        "Inject latency on the app's traffic (SLO-style slowness).",
        ("delete the chaos experiment", "verify latency returns to baseline"),
        template={
            "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "NetworkChaos",
            "metadata": {"name": "{name}", "namespace": "chaos-testing"},
            "spec": {
                "action": "delay", "mode": "all",
                "selector": {"namespaces": ["{namespace}"],
                             "labelSelectors": {"app": "{app}"}},
                "delay": {"latency": "{latency}", "correlation": "100",
                          "jitter": "0ms"},
                "direction": "both", "duration": "{duration}",
            },
        },
    ),
    "network-loss": Scenario(
        "network-loss", "chaos", "PacketLossInjected",
        "Drop a fraction of packets (timeouts, retries, flaky calls).",
        ("delete the chaos experiment", "verify error rate returns to baseline"),
        template={
            "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "NetworkChaos",
            "metadata": {"name": "{name}", "namespace": "chaos-testing"},
            "spec": {
                "action": "loss", "mode": "all",
                "selector": {"namespaces": ["{namespace}"],
                             "labelSelectors": {"app": "{app}"}},
                "loss": {"loss": "{loss}", "correlation": "100"},
                "direction": "both", "duration": "{duration}",
            },
        },
    ),
    "dns-failure": Scenario(
        "dns-failure", "chaos", "DNSResolutionFailure",
        "Break DNS responses for the target (service discovery outage).",
        ("delete the chaos experiment", "verify name resolution recovers"),
        template={
            "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "NetworkChaos",
            "metadata": {"name": "{name}", "namespace": "chaos-testing"},
            "spec": {
                "action": "netem", "mode": "all",
                "selector": {"namespaces": ["{namespace}"],
                             "labelSelectors": {"app": "{app}"}},
                "netem": {}, "duration": "{duration}",
                "externalTargets": ["{target}"],
            },
        },
    ),
    "cpu-stress": Scenario(
        "cpu-stress", "chaos", "CPUSaturation",
        "Saturate CPU on target pods (throttling, missed probes).",
        ("delete the chaos experiment", "verify CPU returns to baseline"),
        template={
            "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "StressChaos",
            "metadata": {"name": "{name}", "namespace": "chaos-testing"},
            "spec": {
                "mode": "one",
                "selector": {"namespaces": ["{namespace}"],
                             "labelSelectors": {"app": "{app}"}},
                "stressors": {"cpu": {"workers": 1, "load": "{load}"}},
                "duration": "{duration}",
            },
        },
    ),
    "memory-stress": Scenario(
        "memory-stress", "chaos", "OOMKilled",
        "Force the container past its memory limit (OOMKilled, exit 137).",
        ("raise or correct the memory limit", "verify pod is running"),
        template={
            "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "StressChaos",
            "metadata": {"name": "{name}", "namespace": "chaos-testing"},
            "spec": {
                "mode": "one",
                "selector": {"namespaces": ["{namespace}"],
                             "labelSelectors": {"app": "{app}"}},
                "stressors": {"memory": {"workers": 1,
                                         "size": "{size}"}},
                "duration": "{duration}",
            },
        },
    ),
    "io-latency": Scenario(
        "io-latency", "chaos", "StorageLatencyInjected",
        "Inject filesystem latency on the pod's volume path.",
        ("delete the chaos experiment", "verify io latency recovers"),
        template={
            "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "IOChaos",
            "metadata": {"name": "{name}", "namespace": "chaos-testing"},
            "spec": {
                "action": "latency", "mode": "one",
                "selector": {"namespaces": ["{namespace}"],
                             "labelSelectors": {"app": "{app}"}},
                "volumePath": "{volume_path}", "path": "{path}",
                "delay": "{delay}", "percent": "100",
                "duration": "{duration}",
            },
        },
    ),
    # ---------- manifest family (Kubernetes-native failures) ----------
    "bad-image-tag": Scenario(
        "bad-image-tag", "manifest", "ImagePullBackOff",
        "Deployment references an image tag that does not exist.",
        ("roll back the deployment", "set a correct image tag"),
        manifest_patch={
            "path": "spec.template.spec.containers[0].image",
            "value": "registry.local/{app}:v{bad_version}",
            "expected_event": "Failed to pull image",
        },
    ),
    "bad-command": Scenario(
        "bad-command", "manifest", "CrashLoopBackOff",
        "Container command exits immediately; restart loop with exit code 0.",
        ("roll back the deployment", "correct the container command"),
        manifest_patch={
            "path": "spec.template.spec.containers[0].command",
            "value": ["/bin/sh", "-c", "exit 0"],
            "expected_event": "CrashLoopBackOff",
        },
    ),
    "missing-env": Scenario(
        "missing-env", "manifest", "CrashLoopBackOff",
        "App crashes on boot: required env var absent (exit code 1).",
        ("restore the missing env var", "roll back the deployment"),
        manifest_patch={
            "path": "spec.template.spec.containers[0].envRemove",
            "value": "{env_key}",
            "expected_event": "CrashLoopBackOff",
        },
    ),
    "probe-mismatch": Scenario(
        "probe-mismatch", "manifest", "LivenessProbeFailure",
        "Liveness probe path returns 404; kubelet restarts healthy pods.",
        ("correct the probe path", "roll back the deployment"),
        manifest_patch={
            "path": "spec.template.spec.containers[0].livenessProbe.httpGet.path",
            "value": "/healthz-wrong",
            "expected_event": "Liveness probe failed",
        },
    ),
    "quota-exceeded": Scenario(
        "quota-exceeded", "manifest", "ResourceQuotaExhausted",
        "Namespace quota already consumed; new pods rejected at admission.",
        ("free quota (scale down other workloads)", "raise the ResourceQuota"),
        manifest_patch={
            "path": "spec.template.spec.containers[0].resources.limits.memory",
            "value": "{oversize}",
            "expected_event": "exceeded quota",
        },
    ),
    "pvc-mismatch": Scenario(
        "pvc-mismatch", "manifest", "PVCMountFailure",
        "Volume references a PersistentVolumeClaim that does not exist.",
        ("create the missing PVC", "fix the claim name"),
        manifest_patch={
            "path": "spec.template.spec.volumes[0].persistentVolumeClaim.claimName",
            "value": "{bad_claim}",
            "expected_event": "references non-existent PVC",
        },
    ),
}


def render_scenario_yaml(name: str, **params) -> str:
    """Render a chaos-family scenario to Chaos Mesh YAML. Manifest-family
    scenarios are patches, not CRs — they render as a JSON spec instead."""
    import json as _json

    sc = SCENARIOS[name]
    if sc.family == "manifest":
        return _json.dumps(sc.manifest_patch, indent=2)
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "pyyaml is required to render chaos scenarios: pip install pyyaml"
        ) from e

    def _fill(obj):
        if isinstance(obj, str):
            for k, v in params.items():
                obj = obj.replace("{" + k + "}", str(v))
            return obj
        if isinstance(obj, dict):
            return {k: _fill(v) for k, v in obj.items()}
        return obj

    defaults = {"name": f"ladder-{name}", "namespace": "default",
                "app": "shop", "duration": "60s"}
    return yaml.dump(_fill(sc.template), sort_keys=False)
