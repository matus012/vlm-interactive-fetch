"""0b client (runs with the python inside vllm.sif): 3 tool-call checks against the local vLLM.

    python hpc/vlm_check.py <base_url> <photo.png> <out.json> [sim_steps.jsonl]   key from env VLLM_API_KEY
"""
from __future__ import annotations

import base64
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.tools import TOOLS  # noqa: E402
from agent.validate import validate_call  # noqa: E402
from openai import OpenAI  # noqa: E402

SYSTEM = (
    "You control a mobile robot with an arm and a mast camera. Each step, pick exactly one skill by calling exactly "
    "one tool. Refer to objects, places and surfaces by name only (e.g. 'red box', 'kitchen table'). Never output "
    "coordinates, poses or numbers for positions. Do not call more than one tool per step."
)
TASK = "Bring the red box to the kitchen table."
HISTORIES = ["Step 1 of 20. History: none. Last result: none.",
             "Step 2 of 20. History: look() -> 4 photos. Last result: the red box is visible on the floor.",
             "Step 3 of 20. History: look(); locate(red box). Last result: found: red box in hall, on floor."]


def main() -> int:
    base, photo, out = sys.argv[1:4]
    simlog = sys.argv[4] if len(sys.argv) > 4 else None
    client = OpenAI(base_url=base, api_key=os.environ["VLLM_API_KEY"], timeout=300)
    model = client.models.list().data[0].id
    url = "data:image/png;base64," + base64.b64encode(Path(photo).read_bytes()).decode()
    calls = []
    for i, hist in enumerate(HISTORIES):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": f"Task: {TASK}\n{hist}\nPhoto 1 (mast camera):"},
                                             {"type": "image_url", "image_url": {"url": url}}]}]
        t0 = time.time()
        r = client.chat.completions.create(model=model, messages=msgs, tools=TOOLS, tool_choice="auto",
                                           temperature=0.0, max_tokens=4096)
        t1 = time.time()
        msg = r.choices[0].message
        tcs = msg.tool_calls or []
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
        rec = {"i": i, "t_start": t0, "t_end": t1, "latency_s": round(t1 - t0, 2), "tool_call_returned": bool(tcs),
               "n_tool_calls": len(tcs), "reasoning_chars": len(reasoning), "content": msg.content,
               "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens}
        if tcs:
            rec["call"] = {"name": tcs[0].function.name, "arguments": tcs[0].function.arguments}
            ok, why = validate_call(tcs[0].function.name, tcs[0].function.arguments)
        else:
            ok, why = False, "no tool call in response"
        if len(tcs) > 1:
            ok, why = False, f"{len(tcs)} tool calls, expected exactly 1"
        rec["valid"], rec["verdict"] = ok, why
        calls.append(rec)
        print(json.dumps(rec), flush=True)
    res = {"model": model, "calls": calls, "valid_count": sum(c["valid"] for c in calls),
           "pass": all(c["valid"] for c in calls), "median_latency_s": statistics.median(c["latency_s"] for c in calls)}
    if simlog and Path(simlog).exists():
        pts = [json.loads(x) for x in Path(simlog).read_text().splitlines() if x]

        def rate(a: float, b: float) -> float | None:
            w = [p for p in pts if a <= p["t"] <= b]
            return round((w[-1]["steps"] - w[0]["steps"]) / (w[-1]["t"] - w[0]["t"])) if len(w) > 1 else None
        res["sim_steps_per_s_during_calls"] = rate(calls[0]["t_start"], calls[-1]["t_end"])
    Path(out).write_text(json.dumps(res, indent=1))
    print("PASS" if res["pass"] else "FAIL", res["valid_count"], "/ 3")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
