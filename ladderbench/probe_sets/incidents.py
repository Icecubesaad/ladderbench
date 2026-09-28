"""incidents@v0-static — 8 static incident seeds (MCQ).

Placeholders until the Chaos Mesh sandbox generates live episodes; each item
presents a short telemetry snippet and asks for the root-cause class. These
exercise the domain head of the model without needing a cluster, and give
the incidents tier something to measure on day one.
"""

from . import Probe

PROBES: list[Probe] = [
    Probe(
        "i01", "incident", "mcq",
        "A pod restarts repeatedly. kubectl describe shows: State: Waiting, "
        "Reason: CrashLoopBackOff, Last State: Terminated, Exit Code: 0, and "
        "logs show only the container entrypoint starting then exiting. "
        "Which root cause class fits best? A) OOM kill B) Process exits "
        "immediately (bad command or script) C) Image pull failure "
        "D) DNS failure",
        "B",
    ),
    Probe(
        "i02", "incident", "mcq",
        "A pod is Terminated with Exit Code 137 and Reason: OOMKilled. "
        "Which root cause class? A) Liveness probe failure B) Bad image tag "
        "C) Container exceeded its memory limit D) Node disk pressure",
        "C",
    ),
    Probe(
        "i03", "incident", "mcq",
        "Pod events show: Failed to pull image 'registry.local/app:v9': "
        "manifest unknown, Back-off pulling image. Which root cause class? "
        "A) Image tag does not exist B) OOM kill C) Readiness probe failure "
        "D) Quota exceeded",
        "A",
    ),
    Probe(
        "i04", "incident", "mcq",
        "Pod events: MountVolume.SetUp failed for volume 'data': references "
        "non-existent PVC 'cache-claim' in namespace 'shop'. Container is "
        "ContainerCreating, never Running. Which root cause class? "
        "A) DNS failure B) PVC not found (volume claim mismatch) "
        "C) CrashLoopBackOff D) CPU throttling",
        "B",
    ),
    Probe(
        "i05", "incident", "mcq",
        "An app pod cannot reach 'orders.shop.svc.cluster.local'; nslookup "
        "from another pod in the same namespace resolves it. Events show "
        "repeated connect timeouts only for this pod. Which class? "
        "A) Cluster DNS outage B) Network policy or per-pod network fault "
        "C) Image pull failure D) Memory limit",
        "B",
    ),
    Probe(
        "i06", "incident", "mcq",
        "Deployment rollout is stuck; replicaset events show: pod "
        "exceeded quota: 'forbidden: exceeded quota: compute-resources, "
        "requested: limits.memory=2Gi'. Which root cause class? "
        "A) ResourceQuota exhaustion B) Liveness probe failure "
        "C) Node not ready D) Bad container command",
        "A",
    ),
    Probe(
        "i07", "incident", "mcq",
        "A pod restarts every ~40s. Previous pod logs show the app serving "
        "traffic fine, then 'Killing container' with reason: liveness probe "
        "failed: HTTP 503 on /healthz during a slow GC pause. Which class? "
        "A) OOM kill B) Image pull failure C) Liveness probe timeout too "
        "aggressive D) PVC mount failure",
        "C",
    ),
    Probe(
        "i08", "incident", "mcq",
        "Several pods across namespaces suddenly move to 'Evicted'. Node "
        "conditions show DiskPressure=True and kubelet logs reference "
        "imagefs usage above the eviction threshold. Which root cause class? "
        "A) Quota exceeded B) Node disk pressure eviction C) DNS failure "
        "D) CrashLoopBackOff",
        "B",
    ),
]
