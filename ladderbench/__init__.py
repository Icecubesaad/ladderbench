"""ladderbench — does fine-tuning break the reasoning-effort dial?

LadderBench scores the *ladder integrity* of any hybrid thinking model served
behind an OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama, ...). It probes
the model at every ``reasoning_effort`` level, then checks four properties:

* token monotonicity   -- lower effort should mean fewer thinking tokens
* accuracy monotonicity -- more effort should not hurt accuracy
* calibration delta    -- per-level distance from a reference (base) curve
* collapse detection   -- empty/absent thinking traces where thinking is expected

Usage (no config files, everything is Python):

    python -m ladderbench score --endpoint http://localhost:8000/v1 \
        --model Qwen/Qwen3.8-27B --probe-set core --out base_curve.json
    python -m ladderbench score --endpoint http://localhost:8000/v1 \
        --model my-finetune --probe-set core --base-curve base_curve.json
    python -m ladderbench probe-sets
    python -m ladderbench selftest

Optional dependencies: ``requests`` (score), ``matplotlib`` (ladder plot),
``pyyaml`` (ladderbench.sandbox scenario rendering). The selftest runs with
none of them.

Project: LADDER. Phase 0 artifact. Probe sets are versioned; core is v1.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
