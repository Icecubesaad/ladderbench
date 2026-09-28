"""Project LADDER on Modal — run the Phase-0/1 study on an H200.

Steps (each is its own CLI so failures are resumable):

    modal run modal_ladder.py::download_cli                 # CPU: 55GB -> volume
    modal run modal_ladder.py::gen_cli                      # H200: generate + filter SFT data
    modal run modal_ladder.py::train_cli --arm arm1         # H200: outcome-only SFT
    modal run modal_ladder.py::train_cli --arm arm2         # H200: reasoning-mixed SFT
    modal run modal_ladder.py::score_cli --model base --out out/base.json
    modal run modal_ladder.py::score_cli --model arm1 --base base --out out/arm1.json
    modal run modal_ladder.py::score_cli --model arm2 --base base --out out/arm2.json

Design notes:
* Data: rejection sampling with the BASE model itself (self-generated SFT).
  Both arms train on the same prompts; arm2 targets contain a <think> trace,
  arm1 targets are direct answers only — the data composition IS the study.
* Scoring uses vLLM's offline batch chat (no HTTP), passing
  chat_template_kwargs={"reasoning_effort": level} — the same mechanism the
  docs prescribe for llama-server. If the dial is not transmitted, levels
  come out identical and the base curve shows it.
* All persistent state lives in the `ladder-cache` Modal volume.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import modal

# ---------------------------------------------------------------- config ----

MODEL_ID = "unsloth/Qwen3.8-27B"  # ungated mirror of Qwen/Qwen3.8-27B
APP_NAME = "ladder"
CACHE = "/cache"
LOCAL_PKG = str(Path(__file__).resolve().parent / "ladderbench")
GPU = "H200"
SEED = 3407

vol = modal.Volume.from_name("ladder-cache", create_if_missing=True)
ENV = {
    "HF_HOME": f"{CACHE}/hf",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "PYTHONPATH": "/root",
    "TOKENIZERS_PARALLELISM": "false",
    # slim image has no nvcc; flashinfer's JIT sampler needs it at runtime
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}

app = modal.App(APP_NAME)

score_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.11.0", "transformers>=5.17", "requests", "pyyaml",
                 "matplotlib", "huggingface_hub")
    .env(ENV)
    .add_local_dir(LOCAL_PKG, "/root/ladderbench")
)
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "pillow", "transformers>=5.17", "trl",
                 "peft", "datasets", "accelerate", "huggingface_hub")
    .env(ENV)
    .add_local_dir(LOCAL_PKG, "/root/ladderbench")
)

# ------------------------------------------------------------ download ----


@app.function(image=train_image, volumes={CACHE: vol}, timeout=3600)
def download_model(model_id: str = MODEL_ID, hf_token: str = "") -> dict:
    from huggingface_hub import snapshot_download

    path = snapshot_download(model_id, token=hf_token or None)
    vol.commit()
    return {"model": model_id, "path": path}


@app.local_entrypoint()
def download_cli(model_id: str = MODEL_ID, hf_token: str = ""):
    print(json.dumps(download_model.remote(model_id, hf_token), indent=2))


@app.function(volumes={CACHE: vol}, timeout=600)
def put_report(report_json: str, name: str) -> dict:
    """Upload a locally-produced report (e.g. base curve) onto the volume so
    other runs can use it for calibration."""
    (Path(CACHE) / "reports").mkdir(parents=True, exist_ok=True)
    (Path(CACHE) / "reports" / f"{name}.json").write_text(report_json,
                                                          encoding="utf-8")
    vol.commit()
    return {"saved": name}


@app.local_entrypoint()
def put_report_cli(file: str, name: str):
    print(json.dumps(put_report.remote(Path(file).read_text(encoding="utf-8"),
                                       name), indent=2))


@app.function(image=train_image, volumes={CACHE: vol}, timeout=1200)
def build_direct_arm(arm: str = "arm3", per_class: int = 140,
                     grounding_n: int = 5000) -> dict:
    """CPU-only dataset build for an outcome-only probe arm: ground-truth
    direct answers over per_class instances + grounding rows. No trace
    generation, no GPU — the collapse-risk composition at scale."""
    from transformers import AutoTokenizer

    from ladderbench.datafactory import (
        SYSTEM_GEN, build_instances, direct_answer, render_training_text,
    )
    from ladderbench.hfdata import load_sft100k_grounding

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    instances = build_instances(seed=SEED, per_class=per_class)
    task_rows = [{"text": render_training_text(
        tok, SYSTEM_GEN, i["question"], direct_answer(i))} for i in instances]
    raw_ground, gstats = load_sft100k_grounding(grounding_n)
    ground_rows = [{"text": tok.apply_chat_template(
        [{"role": "user", "content": r["user"]},
         {"role": "assistant", "content": r["assistant"]}], tokenize=False)}
        for r in raw_ground]
    rows = task_rows + ground_rows
    (Path(CACHE) / "data").mkdir(parents=True, exist_ok=True)
    with open(Path(CACHE) / "data" / f"{arm}.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    vol.commit()
    return {"arm": arm, "task_rows": len(task_rows),
            "grounding_rows": len(ground_rows), "total": len(rows),
            "grounding_stats": gstats}


@app.local_entrypoint()
def build_direct_cli(arm: str = "arm3", per_class: int = 140,
                     grounding_n: int = 5000):
    print(json.dumps(build_direct_arm.remote(arm, per_class, grounding_n),
                     indent=2))


# -------------------------------------------------------- data generation ----


@app.function(image=score_image, gpu=GPU, volumes={CACHE: vol}, timeout=3600)
def gen_data(per_class: int = 4, samples: int = 2, cap: int = 400,
             source: str = "synthetic", grounding_n: int = 0) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from ladderbench.datafactory import (
        SYSTEM_GEN, build_instances, check, direct_answer,
        parse_reply, reasoning_answer, render_training_text, split_think,
    )

    instances = build_instances(seed=SEED, per_class=per_class)
    synthetic_n = len(instances)
    hf_report = None
    if source in ("hf", "mix"):
        from ladderbench.hfdata import load_devops_incident_response
        hf_instances, hf_report = load_devops_incident_response()
        if source == "hf":
            instances = hf_instances
        else:  # mix: synthetic first, real-world appended
            instances = instances + hf_instances
    print(f"[ladder][gen] source={source} instances={len(instances)} "
          f"(synthetic={synthetic_n}, hf={len(instances) - synthetic_n})")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(model=MODEL_ID, dtype="bfloat16", gpu_memory_utilization=0.92,
              max_model_len=8192, seed=SEED)

    messages = [[{"role": "system", "content": SYSTEM_GEN},
                 {"role": "user", "content": inst["question"]}] for inst in instances]
    sp = SamplingParams(n=samples, temperature=0.9, top_p=0.95, max_tokens=2200,
                        seed=SEED)
    outs = llm.chat(messages, sp, chat_template_kwargs={"reasoning_effort": "medium"})

    kept = []          # (instance, trace, final)
    class_ok: dict[str, int] = {}
    no_format, wrong_class, short_trace = 0, 0, 0
    raw_samples: list[str] = []
    parsed_failures: list[str] = []
    for inst, out in zip(instances, outs):
        for comp in out.outputs:
            if len(raw_samples) < 2:
                raw_samples.append(comp.text[:600])
            rc = parse_reply(comp.text)
            if rc is None:
                no_format += 1
                continue
            if not check(rc, inst):
                if len(parsed_failures) < 5:
                    parsed_failures.append(
                        f"{inst['id']}: expected={inst['fault_class']} "
                        f"got={rc[0][:80]!r}")
                wrong_class += 1
                continue
            trace, final = split_think(comp.text)
            if len(trace) < 80:  # need a real trace for the reasoning rows
                short_trace += 1
                continue
            kept.append((inst, trace, final))
            class_ok[inst["fault_class"]] = class_ok.get(inst["fault_class"], 0) + 1
            break  # first correct per instance

    kept = kept[:cap]
    print(f"[ladder][filter] no_format={no_format} wrong_class={wrong_class} "
          f"short_trace={short_trace} kept={len(kept)}")
    for i, raw in enumerate(raw_samples):
        print(f"[ladder][raw-{i}] {raw!r}")
    for f in parsed_failures:
        print(f"[ladder][wrong] {f}")
    for i, (inst, trace, final) in enumerate(kept[:3]):
        print(f"[ladder][kept-{i}] {inst['id']} trace_len={len(trace)} "
              f"final={final[:160]!r}")

    if not kept:
        return {"error": "no generations passed the filter",
                "no_format": no_format, "wrong_class": wrong_class,
                "short_trace": short_trace, "raw_samples": raw_samples}

    # log one rendered example for template verification
    sample_text = render_training_text(tok, SYSTEM_GEN, kept[0][0]["question"],
                                       reasoning_answer(kept[0][1], kept[0][0]))
    print("[ladder][render-sample]\n" + sample_text[:1200])

    arm2_rows, arm1_rows = [], []
    for i, (inst, trace, _final) in enumerate(kept):
        arm2_rows.append({"text": render_training_text(
            tok, SYSTEM_GEN, inst["question"], reasoning_answer(trace, inst))})
        arm1_rows.append({"text": render_training_text(
            tok, SYSTEM_GEN, inst["question"], direct_answer(inst))})
        if i % 3 == 0:  # +25% direct rows in arm2 -> ~75/25 mix
            arm2_rows.append({"text": render_training_text(
                tok, SYSTEM_GEN, inst["question"], direct_answer(inst))})

    import random as _r
    _r.Random(SEED).shuffle(arm2_rows)

    (Path(CACHE) / "data").mkdir(parents=True, exist_ok=True)

    grounding_rows = []
    grounding_stats = None
    if grounding_n > 0:
        from ladderbench.hfdata import load_sft100k_grounding

        raw_rows, grounding_stats = load_sft100k_grounding(grounding_n)
        for r in raw_rows:
            grounding_rows.append({"text": tok.apply_chat_template(
                [{"role": "user", "content": r["user"]},
                 {"role": "assistant", "content": r["assistant"]}],
                tokenize=False)})

    for arm, rows in (("arm1", arm1_rows), ("arm2", arm2_rows)):
        rows = rows + grounding_rows  # Layer B: constant across arms
        with open(Path(CACHE) / "data" / f"{arm}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    vol.commit()

    return {
        "instances": len(instances),
        "kept_correct": len(kept),
        "arm1_rows": len(arm1_rows) + len(grounding_rows),
        "arm2_rows": len(arm2_rows) + len(grounding_rows),
        "grounding_rows": len(grounding_rows),
        "grounding_stats": grounding_stats,
        "by_class": class_ok,
        "render_sample_head": sample_text[:300],
    }


@app.local_entrypoint()
def gen_cli(per_class: int = 4, samples: int = 2, cap: int = 400,
            source: str = "synthetic", grounding_n: int = 0):
    print(json.dumps(gen_data.remote(per_class, samples, cap, source,
                                     grounding_n), indent=2))


# ------------------------------------------------------------ SFT arms ----


@app.function(image=train_image, gpu=GPU, volumes={CACHE: vol}, timeout=21600)
def train_sft(arm: str, epochs: int = 3, max_rows: int = 0) -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    data_path = f"{CACHE}/data/{arm}.jsonl"
    out_dir = f"{CACHE}/checkpoints/{arm}"
    rows = load_dataset("json", data_files=data_path)["train"]
    if max_rows and max_rows < len(rows):
        # budget/scale knob: arm2.jsonl is task rows first, grounding after,
        # so a prefix slice keeps ALL task-core rows and trims grounding only
        rows = rows.select(range(max_rows))
    print(f"[ladder][train:{arm}] {len(rows)} rows from {data_path}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    cfg = SFTConfig(
        output_dir=f"{CACHE}/tmp/{arm}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        max_length=2048,
        dataset_text_field="text",
        seed=SEED,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=rows,
        peft_config=LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
    )
    trainer.train()

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    vol.commit()
    return {"arm": arm, "rows": len(rows), "saved": out_dir,
            "loss_history": [round(s["loss"], 4) for s in trainer.state.log_history
                             if "loss" in s]}


@app.local_entrypoint()
def train_cli(arm: str, epochs: int = 3, max_rows: int = 0):
    print(json.dumps(train_sft.remote(arm, epochs, max_rows), indent=2))


# ------------------------------------------------------------ scoring ----


@app.function(image=score_image, gpu=GPU, volumes={CACHE: vol}, timeout=7200)
def score_model(model_key: str, base_key: str = "") -> dict:
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vllm import LLM, SamplingParams

    from ladderbench.extraction import check_answer, extract_answer
    from ladderbench.metrics import analyze
    from ladderbench.probe_sets import load_probe_set, probe_set_version
    from ladderbench.report import build_report
    from ladderbench.runner import DEFAULT_LEVELS, LevelResult
    from ladderbench.datafactory import split_think

    def resolve(key: str) -> str:
        if key.startswith("hf:"):
            return key[3:]  # any HF repo id, e.g. the ThinkingCap control
        return MODEL_ID if key == "base" else f"{CACHE}/checkpoints/{key}"

    problems = load_probe_set("all")
    messages = [[
        {"role": "system", "content": "Answer the problem. End with the final "
         "numeric answer on its own, formatted as: The answer is N"},
        {"role": "user", "content": p.question},
    ] for p in problems]

    llm = LLM(model=resolve(model_key), dtype="bfloat16",
              gpu_memory_utilization=0.92, max_model_len=16384, seed=SEED)
    sp = SamplingParams(temperature=0.0, max_tokens=4096, seed=SEED)

    levels: dict[str, LevelResult] = {}
    for level in DEFAULT_LEVELS:
        res = LevelResult(level=level)
        outs = llm.chat(messages, sp,
                        chat_template_kwargs={"reasoning_effort": level})
        for probe, out in zip(problems, outs):
            comp = out.outputs[0]
            trace, final = split_think(comp.text)
            tokens = max(1, (len(trace) + 3) // 4) if trace else 0
            pred = extract_answer(final if final else comp.text, probe.kind)
            res.n += 1
            res.n_correct += int(check_answer(pred, probe.answer))
            res.thinking_tokens.append(tokens)
            res.trace_chars.append(len(trace))
            res.trace_empty += int(len(trace.strip()) < 40)
            if comp.token_ids is not None:
                res.completion_tokens.append(len(comp.token_ids))
            res.responses.append({"id": probe.id, "correct": bool(check_answer(pred, probe.answer)),
                                  "predicted": pred, "trace_chars": len(trace)})
        levels[level] = res
        print(f"[ladder][score:{model_key}] {level}: acc={res.accuracy:.3f} "
              f"med_tokens={res.median_tokens:.0f} empty={res.empty_rate:.0%}")

    base_report = None
    if base_key:
        base_path = Path(CACHE) / "reports" / f"{base_key}.json"
        base_report = json.loads(base_path.read_text(encoding="utf-8"))

    analysis = analyze(levels, DEFAULT_LEVELS,
                       base=base_report["summary"] if base_report else None)
    report = build_report(model_key, "vllm-offline", f"all@{probe_set_version('core')}",
                          levels, analysis, DEFAULT_LEVELS)

    # ladder plot -> png bytes
    fig, ax = plt.subplots(figsize=(8, 6))
    s = report["summary"]
    ax.plot(s["median_thinking_tokens"], s["accuracy"], "o-", lw=2, ms=9,
            label=f"{model_key}")
    for lv, x, y in zip(s["levels"], s["median_thinking_tokens"], s["accuracy"]):
        ax.annotate(lv, (x, y), textcoords="offset points", xytext=(8, -4), fontsize=9)
    if base_report:
        b = base_report["summary"]
        ax.plot(b["median_thinking_tokens"], b["accuracy"], "s--", lw=2, ms=8,
                label=f"{base_key} (base)")
    ax.set_xlabel("median thinking tokens (est.)")
    ax.set_ylabel("probe accuracy")
    ax.set_title(f"Ladder integrity: {model_key} — {report['ladder_integrity']}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)

    (Path(CACHE) / "reports").mkdir(parents=True, exist_ok=True)
    (Path(CACHE) / "reports" / f"{model_key}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    vol.commit()
    return {"report": report, "plot_b64": base64.b64encode(buf.getvalue()).decode()}


@app.local_entrypoint()
def score_cli(model_key: str, base: str = "", out: str = "out/run.json"):
    result = score_model.remote(model_key, base)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result["report"], indent=2), encoding="utf-8")
    plot_path = out_path.with_suffix(".png")
    plot_path.write_bytes(base64.b64decode(result["plot_b64"]))
    rep = result["report"]
    print(f"verdict: {rep['ladder_integrity'].upper()}")
    for f in rep["findings"]:
        print(f"  - {f}")
    print(f"report -> {out_path}\nplot   -> {plot_path}")


# ------------------------------------------------------- domain validation ----
# Held-out incident instances (unseen seed), all models x all effort levels:
# generalization, recovery validity, and the dial's behavior ON the domain.

DOMAIN_SEED = 777


@app.function(image=score_image, gpu=GPU, volumes={CACHE: vol}, timeout=7200)
def domain_eval(models: str = "base,arm1,arm2", train_per_class: int = 40) -> dict:
    from vllm import LLM, SamplingParams

    from ladderbench.datafactory import (
        SYSTEM_GEN, build_instances, check, parse_reply, split_think,
    )
    from ladderbench.runner import DEFAULT_LEVELS

    # train_q must reproduce EXACTLY the instances gen was run with
    # (seed + per_class), otherwise low-cardinality scenarios — whose question
    # space is only ~80-240 distinct strings — can leak synthetic twins from
    # training into this eval.
    train_q = {i["question"] for i in build_instances(seed=SEED,
                                                      per_class=train_per_class)}
    held = [i for i in build_instances(seed=DOMAIN_SEED, per_class=2)
            if i["question"] not in train_q]
    messages = [[{"role": "system", "content": SYSTEM_GEN},
                 {"role": "user", "content": inst["question"]}] for inst in held]
    print(f"[ladder][domain] {len(held)} held-out instances")

    def recovery_valid(rec: str, expected: str) -> bool:
        key = expected.split()[0].lower().strip(".,")  # e.g. 'delete', 'restart'
        return key in rec.lower()

    results: dict = {"held_out_instances": len(held), "seed": DOMAIN_SEED,
                     "models": {}}
    for model_key in models.split(","):
        path = MODEL_ID if model_key == "base" else f"{CACHE}/checkpoints/{model_key}"
        llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=0.92,
                  max_model_len=16384, seed=SEED)
        sp = SamplingParams(temperature=0.0, max_tokens=2200, seed=SEED)
        per_level: dict = {}
        for level in DEFAULT_LEVELS:
            outs = llm.chat(messages, sp,
                            chat_template_kwargs={"reasoning_effort": level})
            n = n_ok = n_valid = trace_sum = 0
            for inst, out in zip(held, outs):
                text = out.outputs[0].text
                rc = parse_reply(text)
                trace, _final = split_think(text)
                n += 1
                if rc and check(rc, inst):
                    n_ok += 1
                    n_valid += int(recovery_valid(rc[1], inst["expected_action"]))
                trace_sum += len(trace)
            per_level[level] = {
                "accuracy": round(n_ok / n, 4),
                "recovery_validity": round(n_valid / n, 4),
                "median_trace_chars": int(sorted(
                    [len(split_think(o.outputs[0].text)[0]) for o in outs])[n // 2]),
            }
            print(f"[ladder][domain:{model_key}] {level}: acc={per_level[level]['accuracy']} "
                  f"valid={per_level[level]['recovery_validity']} "
                  f"trace={per_level[level]['median_trace_chars']}")
        results["models"][model_key] = per_level
        del llm
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()

    (Path(CACHE) / "reports" / "domain_eval.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    vol.commit()
    return results


@app.local_entrypoint()
def domain_cli(models: str = "base,arm1,arm2", out: str = "out/domain_eval.json",
               train_per_class: int = 40):
    result = domain_eval.remote(models, train_per_class)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"domain eval -> {out_path}")
