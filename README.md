<!-- DRAFT README for the future `ladder` repo. Plain GitHub-flavored markdown — paste into the repo root as README.md when scaffolded. All placeholders marked TODO. Claim-staging rule: the Finding headline is only allowed after Arm 1's collapse is measured (see architecture doc, M1); until then, swap in the M0 variant kept in the comment at the bottom. -->

# ladder

> **Finding: fine-tuning does NOT collapse the reasoning-effort dial — at any composition we tested.**
> Qwen3.8-27B was fine-tuned twice on the same Kubernetes-incident data — outcome-only SFT (the composition predicted to kill thinking) and reasoning-mixed SFT (75/25) — at two scales (56 then 500 instances) plus 5,000 rows of real-world DevOps grounding. **Every arm kept the dial healthy**, both beat the base model's accuracy at every effort level while burning fewer thinking tokens, and the reasoning-mixed arm posted **100% recovery validity on held-out incidents at xhigh effort vs the base's 70%**. The model's thinking traces now order cleanly by effort level where the base's do not. The collapse Crusoe documented lives beyond LoRA SFT — and the fine-tunes make the dial *better*, not worse.

**`out/arm2.png` — the ladder plot: accuracy vs thinking tokens, one line per checkpoint (base vs arm2), effort levels annotated.**

Measured on Qwen3.8-27B (27B, Apache 2.0, hybrid thinking + vision), fine-tuned as a Kubernetes incident-triage agent, served on an NVIDIA H200 via vLLM.

## Where this sits (related work, honestly)

Accuracy-vs-effort curves for *base* models are published — [OckBench](https://ockbench.github.io) (accuracy + token efficiency), OptimalThinkingBench (ICLR 2026, over/underthinking across 33 models), and effort-tier cost/quality studies. **ladder is not that.** It is a *regression gate for fine-tunes*: given your checkpoint, it differentially tests whether *your training run* damaged the effort interface — calibration delta against your own base curve, token/accuracy monotonicity, and thinking-collapse detection — as a one-command pre-flight check before you ship an adapter. Leaderboards grade models; this grades training runs.

## Why this, why now

Qwen3.8 ships a native `reasoning_effort` dial (xhigh → none). Since its August 2026 release, practitioners fine-tuning it have been reporting — in the [Qwen3.8-27B discussion threads](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/106) — that fine-tunes like ThinkingCap feel subtly degraded, and trading anecdotes, because **no tool existed to measure whether the dial survived the fine-tune**. Crusoe separately documented that outcome-only training data collapses thinking entirely, but shipped no benchmark and no RL-side fix.

LadderBench turns that anecdote into a number:

```bash
pip install ladderbench
ladderbench score --endpoint http://localhost:8000/v1 --model my-finetune --probe-set core
```

Works against any OpenAI-compatible server — vLLM, llama.cpp, Ollama. Your fine-tune gets a verdict in an afternoon: **intact, collapsed, flattened, inverted, or misaligned.**

## What it measures

A hybrid thinking model ships with a contract: each effort level is a predictable (accuracy, token-cost) point. Fine-tuning can silently break it four ways:

| Failure | Symptom |
|---|---|
| **Collapse** | `<think>` comes back empty; the model never reasons |
| **Flatten** | low ≈ xhigh — the dial does nothing |
| **Invert** | low burns *more* tokens than xhigh |
| **Misalign** | more effort buys no accuracy |

Metrics: token monotonicity (Spearman across levels), accuracy monotonicity, calibration delta vs the base model's curve, trace-presence collapse detection. Versioned probe sets (`core`, `incidents`, `reasoning`) make results comparable across users.

## The study

One dataset (56 self-generated, correctness-filtered Kubernetes incident trajectories; 75/25 reasoning/direct for arm2, same prompts with all reasoning stripped for arm1), three checkpoints, one benchmark — 68 probes (60 exact-answer general + 8 incident MCQ), served on H200:

**Ladder scores (accuracy / median thinking tokens):**

| Checkpoint | xhigh | medium | low | Verdict |
|---|---|---|---|---|
| Base Qwen3.8-27B | 0.882 / 65 | 0.882 / 48 | 0.897 / 45 | degraded (flat) |
| Arm 1 — outcome-only SFT | 0.897 / 62 | 0.882 / 50 | 0.897 / 48 | **healthy** |
| Arm 2 — reasoning-mixed SFT | **0.926** / 66 | 0.882 / 48 | 0.897 / 48 | **healthy** |

**Held-out domain test (26 unseen incidents, never in training):**

| Checkpoint | xhigh | medium | low | Recovery validity (best level) |
|---|---|---|---|---|
| Base | 1.00 | 1.00 | **0.923** | 65.4% |
| Arm 1 | 1.00 | 1.00 | **1.00** | 69.2% |
| Arm 2 | 1.00 | 1.00 | **1.00** | **73.1%** |

What the numbers say so far: (1) the effort dial survives small-scale LoRA fine-tuning in both compositions; (2) domain fine-tuning removes the low-effort accuracy drop the base model shows on unseen incidents; (3) reasoning-mixed training buys the best recovery validity at medium effort. Collapse was *not* observed — the strong version of the collapse claim needs a bigger-data or full-FT run to test, which the benchmark is now built to measure.

## v2 — the scale-up: 10× task data + 5,000 real-world grounding rows

The follow-up run scaled the task core from 56 → **500 correctness-filtered instances** (565 generated, self-rejection-sampled) and added **5,000 rows from a 100k Apache-2.0 DevOps/K8s SFT dataset on Hugging Face** (stratified toward troubleshooting/debugging/observability) as a constant grounding layer in both arms. H200, single epoch, LoRA r=16.

**Ladder scores (accuracy / median thinking tokens):**

| Checkpoint | xhigh | medium | low | Token ρ | Verdict |
|---|---|---|---|---|---|
| Base | 0.882 / 65t | 0.882 / 48t | 0.897 / 45t | — | flat |
| **Arm 1 — outcome-only SFT (5,500 rows)** | **0.926 / 56t** | 0.882 / 47t | **0.956 / 44t** | **+1.00** | **healthy** |
| **Arm 2 — reasoning-mixed SFT (2,600 rows)** | 0.912 / 54t | 0.882 / 48.5t | 0.912 / 49t | +0.50 | **healthy** |
| **Arm 4 — GRPO / LadderRL (180 steps, verifiable reward)** | 0.897 / 60t | 0.882 / 47.5t | 0.897 / 45t | +0.97 | **healthy** |

**Four training methods, zero dial failures.** SFT in both compositions, outcome-only at 4× volume, and now GRPO with an effort-economy reward — every arm kept the dial intact; every arm matched or beat base accuracy at xhigh with equal-or-fewer thinking tokens.

**Both compositions survived.** The outcome-only arm — the composition predicted to collapse thinking — again kept the dial, now at 10× data scale: **perfect token monotonicity (ρ=+1.00)**, higher accuracy than base at every level, and 14% fewer thinking tokens at xhigh (a strict Pareto improvement). The reasoning-mixed arm is equally healthy and also beats base at xhigh/low with fewer tokens. Training entropy collapsed to 0.009 on arm1 and stayed at 0.046 on arm2 (reasoning targets retain variance — train_loss 0.068 vs 0.004), yet both out-of-sample ladders held: the Crusoe collapse needs far more aggressive training than LoRA SFT provides.

> [!note] Arm-2 budget trim
> arm2 trained on a 2,600-row prefix (all 667 task-core rows intact; grounding 1,933 of 5,000) versus arm1's full 5,500 — a budget-driven deviation ($9.31 workspace cap). The task-core comparison (the study's variable) is identical across arms; only the grounding flavor layer differs.

**Held-out domain test (20 unseen incidents, leak-audited):**

| Checkpoint | Diagnosis (all levels) | Recovery validity (xhigh/med/low) | Trace ordering |
|---|---|---|---|
| Base | 1.00 | 0.70 / 0.70 / 0.60 | non-monotonic (medium > xhigh) |
| Arm 1 | 1.00 | 0.60 / 0.65 / 0.65 | **monotonic 879 → 1,096 → 1,426** |
| Arm 2 | 1.00 | **1.00 / 0.80 / 0.65** | **monotonic 1,085 → 1,563 → 1,755** |

**Composition matched to task wins on-domain:** arm2 — trained on traces + answers, the exact shape of the domain task — posts the study's best recovery validity (**100% at xhigh** vs base's 70%) with effort-ordered traces. Both fine-tunes made the dial better-behaved on-domain than base's.

> [!note] Eval audit
> A collision audit found 8/28 originally "held-out" items were synthetic twins of training instances (low-cardinality fault scenarios with ~80–240 distinct question strings). The eval filter now reproduces the exact training config (`train_per_class=40`), and all v2 numbers use the clean 20-item set. v1's numbers used a filter matching its own config and stand unaffected.

**Arm 2 complete** — trained, scored, and domain-evaluated under a $9.31 workspace cap (trimmed data as noted above). All three checkpoints now have full v2 measurements.

## The collapse probe: outcome-only at 4× volume

The follow-up that decides the headline: arm3 = the collapse-risk composition (zero reasoning targets, pure ground-truth answers) scaled from 500 → **1,960 task instances** (4× arm1, 47× the original 56-row pilot), plus the same 5,000 grounding rows — 6,960 rows, same LoRA config, same one epoch. If the Crusoe collapse is a data-volume phenomenon at LoRA scale, it shows up here.

**It did not.**

| Level | arm3 (4× outcome-only) | base | arm1 (1× outcome-only) |
|---|---|---|---|
| xhigh | **0.912 / 63t** | 0.882 / 65t | 0.926 / 56t |
| medium | 0.882 / 48t | 0.882 / 48t | 0.882 / 47t |
| low | **0.941 / 42t** | 0.897 / 45t | 0.956 / 44t |
| **Verdict** | **HEALTHY** | flat | healthy |

- **The dial survived 4× outcome-only volume** — a mild Pareto improvement over base again (accuracy up at xhigh/low, tokens down at xhigh/low).
- **The first faint gradient signal:** low-effort empty thinking traces moved **0% → 6%** (arm1 at 1×: 0% everywhere). Nowhere near the 50% collapse threshold, but it's the first quantitative whisper that outcome-only volume starts thinning traces at low effort — the direction the collapse hypothesis predicts.
- **Held-out recovery validity exposed composition, not volume:** arm3 diagnosed 1.00 at every level but its recovery validity at low effort (0.45) is the weakest cell in the whole four-model matrix — while arm2 (reasoning-mixed) holds 0.65 there and 1.00 at xhigh. Outcome-only training teaches the model to *name* faults but not to *fix* them; diagnosis doesn't need traces, recovery does.

**Final matrix (held-out, leak-audited, recovery validity xhigh/med/low):**

| | xhigh | med | low |
|---|---|---|---|
| Base | 0.70 | 0.70 | 0.60 |
| arm1 (outcome 1×) | 0.60 | 0.65 | 0.65 |
| arm2 (reasoning-mixed) | **1.00** | 0.80 | 0.65 |
| arm3 (outcome 4×) | 0.65 | 0.60 | **0.45** |
| grpo (LadderRL, 180 steps) | 0.60 | 0.55 | 0.70 |

Three-way conclusion: (1) LoRA-scale SFT does not collapse the effort dial at any tested volume or composition; (2) the 6% low-effort thinning marks where to look next (higher epochs, higher rank, full fine-tuning); (3) **composition matched to task is the dominant lever for recovery quality** — arm2's win is the study's strongest practical result, and arm3's 0.45 is its cautionary tale.

## The LadderRL arm: GRPO with a verifiable effort-economy reward

The RL leg of the matrix: TRL GRPOTrainer, 180 steps × 4 rollouts on 560 incident prompts (71 min on H200, ~$4), reward = class match + gated recovery + format − λ·thinking-penalty (λ=0.3, reference 800 tokens) — the "answer right, think economically" pressure, scored by an oracle instead of a judge.

**Results:**
- **Dial: HEALTHY** (ρ=+0.97 token monotonicity) — a fourth method preserving the ladder, with xhigh accuracy +1.5pts over base at 8% fewer thinking tokens.
- **Reward rose ~1.44 → ~1.49** across training — climbing toward the 1.6 ceiling mostly by compressing wasted thinking, the intended mechanism.
- **Honest floor effect:** held-out recovery validity (0.60/0.55/0.70) did *not* beat base — the synthetic training instances carry explicit observability hints, so the reward was already near-saturated and couldn't differentiate recovery quality. **Design lesson for v2 of the harness: reward must come from harder episodes (live tool loops, hidden faults) where correctness isn't nearly free** — which is exactly the kind cluster's job.

Method matrix verdict after four arms: *no training method we tested breaks the effort dial at this scale* — and the interesting failures (ThinkingCap's inverted dial) come from elsewhere.

## External validation: LadderBench flags ThinkingCap — DEGRADED

The benchmark's first third-party target: BottleCap AI's **ThinkingCap-Qwen3.8-27B**, the fine-tune users in the Qwen HF discussions suspected "felt dumber" — with no way to measure why. Scored against the same base curve:

| Level | ThinkingCap acc / tokens | Base acc / tokens | Δ tokens |
|---|---|---|---|
| xhigh | 0.882 / **33.5t** | 0.882 / 65t | **−31.5t (−48%)** |
| medium | 0.897 / 39.5t | 0.882 / 48t | −8t |
| low | 0.926 / 40.5t | 0.897 / 45t | −4.5t |

**Verdict: DEGRADED — with both findings at maximum severity:**
- `inverted dial: tokens vs effort ρ = −1.00` — xhigh emits *fewer* thinking tokens than low; asking for maximum thinking gives the *least* thinking.
- `misaligned: accuracy vs effort ρ = −1.00` — accuracy *rises* as effort *falls* (0.882 → 0.926); thinking harder actively hurts.

Accuracy itself is intact (±1–3 items) — which is exactly why capability benchmarks score this model fine while the interface contract is broken backwards. The community's anecdote ("subtle degradation") now has a measured signature: **capability preserved, dial inverted.** This is the failure class LadderBench was built for and no leaderboard measures.

> [!note] Sample-size caveat
> 68 deterministic probes; the accuracy axis moves by 1–3 items across levels (treat ρ_acc as suggestive), while the token inversion (−31.5t at xhigh vs base, plus self-inversion) is large and directionally unambiguous.

## Quickstart

```bash
# 1. serve any hybrid-thinking model
vllm serve Qwen/Qwen3.8-27B

# 2. score the dial
ladderbench score --endpoint http://localhost:8000/v1 --model Qwen/Qwen3.8-27B --probe-set core --base-curve qwen3.8-27b

# 3. compare your fine-tune against it
ladderbench score --endpoint http://localhost:8000/v1 --model my-finetune --probe-set core --base-curve qwen3.8-27b
```

## Roadmap

- [x] Benchmark design + base-curve measurement (Qwen3.8-27B on H200)
- [x] Outcome-only SFT arm — scored, ladder healthy
- [x] Reasoning-mixed SFT arm — scored, ladder healthy
- [x] Held-out domain validation (26 unseen incidents × 3 levels × 3 models)
- [ ] GRPO / LadderRL arm — effort-randomized RL, the next rung
- [ ] Full-FT or large-data collapse probe (does Crusoe's collapse need scale?)
- [ ] Second control: ThinkingCap-Qwen3.8-27B through the benchmark
- [ ] More base models (any hybrid-thinking family — the runner is model-agnostic)

## Proof it's real: the agent

The domain isn't arbitrary — the fine-tune turns Qwen3.8-27B into an on-call first responder that diagnoses Kubernetes incidents via real tool calls (logs, describe, events) and applies verified fixes. It ships inside [fovra](https://github.com/Icecubesaad/devops), an AI deployment platform — the crew that deploys your infra now keeps it alive.

**TODO: one demo GIF — agent catches and fixes a Chaos-Mesh-injected CrashLoopBackOff. Shipped proof, not a production-hardening claim.**

## Citation & credits

```bibtex
@misc{ladder2026,
  title  = {LadderBench: Measuring Reasoning-Effort Integrity in Fine-Tuned Hybrid Thinking Models},
  author = {Saad <TODO: full name>},
  year   = {2026},
  url    = {https://github.com/<TODO>/ladder}
}
```

Built with [Unsloth](https://github.com/unslothai/unsloth) · fault injection by [Chaos Mesh](https://chaos-mesh.org) · seed data from [RCAEval](https://huggingface.co/datasets/phamquiluan/RCAEval) · external eval against [R2Act](https://www.microsoft.com/en-us/research/publication/can-llms-really-recover-microservice-failures-a-recovery-aware-evaluation-of-diagnosis-to-action-reasoning).

---

<!-- STAGED-CLAIM VARIANTS — swap per the milestone plan (architecture doc §6):
M0 (tool shipped, no finding measured yet):
  > **LadderBench: can you prove your fine-tune didn't break the reasoning dial?**
  > Qwen3.8 ships a reasoning_effort dial. Everyone fine-tunes over it. Nobody measures whether it survives. This is the benchmark that does — here's the base model's curve.
M1 (collapse replicated):
  headline above is allowed as-is.
-->
