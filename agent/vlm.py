"""VLM clients (task 004). Same interface: `vlm(packet, seed) -> Reply`. Stateless per step: the packet carries the
history. Stdlib only at import (the OpenAI client is imported by OpenAIVLM).

OpenAIVLM: OpenAI-compatible chat completions (vLLM), tools = agent.tools.TOOLS, tool_choice "auto"; messages = system
+ one user message (packet text, then "Photo k" + the k-th photo as a base64 PNG data URL). Sampling defaults
(Qwen3 thinking mode): temperature 0.6, top_p 0.95, top_k 20 (extra_body), max_tokens 8192; `seed` per call (the loop
passes task seed x 100 + step). Transient HTTP errors (connection, timeout, 429, 5xx) are retried up to RETRIES times,
each logged in Reply.retries and on stderr; a response that arrived is never retried, whatever it holds.

StubVLM(items): plays a call list, one item per step. Item = {"name", "arguments"} (one call; arguments a dict or a
raw string, e.g. invalid JSON) or a list of such dicts (zero or several calls: a malformed step). Past the end of
the list it returns zero calls.
"""
from __future__ import annotations

import base64
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agent.context import Packet
from agent.tools import TOOLS

RETRIES = 3
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 8192}


@dataclass
class Reply:
    content: str | None
    reasoning: str | None
    tool_calls: list[dict]  # [{"name", "arguments" (JSON string as sent)}]
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    retries: list[str] = field(default_factory=list)

    def as_log(self) -> dict:
        return asdict(self)


def messages(packet: Packet) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": packet.user}]
    for label, path in zip(packet.labels, packet.photos, strict=True):
        assert path is not None, "a real VLM call needs saved photos (SimRobot photo_dir)"
        url = "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()
        content += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": url}}]
    return [{"role": "system", "content": packet.system}, {"role": "user", "content": content}]


class OpenAIVLM:
    def __init__(self, base_url: str, api_key: str, model: str | None = None, timeout_s: float = 300.0,
                 **sampling: float) -> None:
        import openai

        self.openai = openai
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)
        self.model = model or self.client.models.list().data[0].id
        self.sampling = SAMPLING | sampling

    def __call__(self, packet: Packet, seed: int) -> Reply:
        s = dict(self.sampling)
        top_k = s.pop("top_k")
        transient = (self.openai.APIConnectionError, self.openai.APITimeoutError, self.openai.RateLimitError,
                     self.openai.InternalServerError)
        retries: list[str] = []
        msgs = messages(packet)
        while True:
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(model=self.model, messages=msgs, tools=TOOLS,
                                                        tool_choice="auto", seed=seed, extra_body={"top_k": top_k}, **s)
                break
            except transient as e:
                if len(retries) == RETRIES:
                    raise
                retries.append(f"{type(e).__name__}: {e}")
                print(f"vlm retry {len(retries)}/{RETRIES}: {retries[-1]}", file=sys.stderr, flush=True)
                time.sleep(2.0 * len(retries))
        dt = time.time() - t0
        msg = r.choices[0].message
        return Reply(content=msg.content,
                     reasoning=getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None),
                     tool_calls=[{"name": c.function.name, "arguments": c.function.arguments}
                                 for c in msg.tool_calls or []],
                     usage=r.usage.model_dump() if r.usage else {}, latency_s=round(dt, 3), retries=retries)


class StubVLM:
    def __init__(self, items: list[dict | list[dict]]) -> None:
        self.items = list(items)
        self.n = 0

    @classmethod
    def from_file(cls, path: str | Path) -> "StubVLM":
        return cls(json.loads(Path(path).read_text()))

    def __call__(self, packet: Packet, seed: int) -> Reply:
        item = self.items[self.n] if self.n < len(self.items) else []
        self.n += 1
        calls = item if isinstance(item, list) else [item]
        tc = [{"name": c["name"], "arguments": c["arguments"] if isinstance(c.get("arguments"), str)
               else json.dumps(c.get("arguments", {}))} for c in calls]
        return Reply(content=None, reasoning="stub", tool_calls=tc, usage={}, latency_s=0.0)
