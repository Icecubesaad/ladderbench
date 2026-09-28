"""Probe runner: iterates effort levels against an OpenAI-compatible endpoint.

The client speaks the chat-completions API and transmits the effort level two
ways, because servers disagree: ``reasoning_effort`` as a top-level parameter
(vLLM/OpenAI style) and ``chat_template_kwargs: {"reasoning_effort": ...}``
(llama.cpp style). In "auto" mode a 400/422 response triggers a retry without
the variant the server rejected, so one client works against every stack.

Thinking tokens are estimated from the reasoning trace (``reasoning_content``
if the server separates it, else the ``<think>...</think>`` span). No
tokenizer dependency by design: the estimate is chars/4, and raw trace text
lengths are stored alongside so any tokenizer can be applied later.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable

import requests

from .extraction import check_answer, extract_answer

LEVELS = ("xhigh", "high", "medium", "low", "none")  # superset, for custom runs
DEFAULT_LEVELS = ("xhigh", "medium", "low")  # the ladder Qwen3.8's template accepts

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_OPEN = re.compile(r"<think>(.*)", re.DOTALL)  # unterminated trace
CHARS_PER_TOKEN = 4.0


@dataclass
class LevelResult:
    level: str
    n: int = 0
    n_correct: int = 0
    thinking_tokens: list[int] = field(default_factory=list)
    trace_chars: list[int] = field(default_factory=list)
    trace_empty: int = 0
    completion_tokens: list[int] = field(default_factory=list)
    responses: list[dict] = field(default_factory=list)  # id, correct, trace sample

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def median_tokens(self) -> float:
        return _median(self.thinking_tokens)

    @property
    def empty_rate(self) -> float:
        return self.trace_empty / self.n if self.n else 0.0


def _median(xs: list[int]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return float(s[m]) if len(s) % 2 else (s[m - 1] + s[m]) / 2


class EndpointClient:
    """Thin chat-completions client with effort-level transmission."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        kwargs_mode: str = "auto",  # auto | both | openai | llamacpp
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.kwargs_mode = kwargs_mode
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _payload(self, messages, level, temperature, max_tokens, mode) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if mode in ("both", "openai"):
            payload["reasoning_effort"] = level
        if mode in ("both", "llamacpp"):
            payload["chat_template_kwargs"] = {"reasoning_effort": level}
        return payload

    def chat(self, messages, level, temperature=0.0, max_tokens=8192) -> dict:
        modes = {
            "auto": ["both", "openai", "llamacpp"],
            "both": ["both"],
            "openai": ["openai"],
            "llamacpp": ["llamacpp"],
        }[self.kwargs_mode]
        last_error = None
        for mode in modes:
            r = requests.post(
                f"{self.endpoint}/chat/completions",
                headers=self.headers,
                data=json.dumps(self._payload(messages, level, temperature, max_tokens, mode)),
                timeout=self.timeout,
            )
            if r.status_code in (400, 422) and mode != modes[-1]:
                last_error = f"{r.status_code}: {r.text[:300]}"
                continue  # server rejected a kwarg variant; retry narrower
            r.raise_for_status()
            return self._parse(r.json())
        raise RuntimeError(f"endpoint rejected request: {last_error}")

    @staticmethod
    def _parse(body: dict) -> dict:
        msg = body["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if not reasoning:
            m = _THINK.search(content) or _THINK_OPEN.search(content)
            reasoning = m.group(1) if m else ""
        usage = body.get("usage") or {}
        return {
            "content": content,
            "reasoning": reasoning,
            "completion_tokens": usage.get("completion_tokens"),
        }


def _make_messages(question: str) -> list[dict]:
    system = (
        "Answer the problem. End with the final numeric answer on its own, "
        "formatted as: The answer is N"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def run_level(
    client: EndpointClient,
    problems: list,
    level: str,
    samples: int = 1,
    max_tokens: int = 8192,
    transport: Callable | None = None,
) -> LevelResult:
    """Probe every problem once at ``level``. ``transport`` overrides
    ``client.chat`` (used by the selftest to inject a fake model)."""
    chat = transport or client.chat
    res = LevelResult(level=level)
    for probe in problems:
        messages = _make_messages(probe.question)
        for _ in range(samples):
            reply = chat(messages, level, temperature=0.0, max_tokens=max_tokens)
            reasoning = reply.get("reasoning", "")
            tokens = _estimate_tokens(reasoning)
            empty = len(reasoning.strip()) < 40  # near-empty trace
            pred = extract_answer(reply["content"], probe.kind)
            correct = check_answer(pred, probe.answer)
            res.n += 1
            res.n_correct += int(correct)
            res.thinking_tokens.append(tokens)
            res.trace_chars.append(len(reasoning))
            res.trace_empty += int(empty)
            if reply.get("completion_tokens") is not None:
                res.completion_tokens.append(reply["completion_tokens"])
            res.responses.append(
                {
                    "id": probe.id,
                    "correct": correct,
                    "predicted": pred,
                    "trace_chars": len(reasoning),
                }
            )
    return res


def run_probe(
    client: EndpointClient,
    problems: list,
    levels: tuple[str, ...] = DEFAULT_LEVELS,
    samples: int = 1,
    max_tokens: int = 8192,
    transport: Callable | None = None,
) -> dict[str, LevelResult]:
    """Probe every problem at every effort level. Order: high effort first."""
    unknown = [lv for lv in levels if lv not in LEVELS]
    if unknown:
        raise ValueError(f"unknown effort levels: {unknown}; known: {LEVELS}")
    out: dict[str, LevelResult] = {}
    for i, level in enumerate(levels, 1):
        print(f"[ladderbench] probing level {i}/{len(levels)}: {level}")
        out[level] = run_level(client, problems, level, samples, max_tokens, transport)
        print(
            f"[ladderbench]   accuracy={out[level].accuracy:.3f} "
            f"median_think_tokens={out[level].median_tokens:.0f} "
            f"empty_traces={out[level].empty_rate:.0%}"
        )
    return out
