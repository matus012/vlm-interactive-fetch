"""G1-colour client + sim-scaling worker (task 002). Rule, truth and scoring: sim/colour_photos.py docstring.

    python hpc/colour_check.py vlm <base_url> <photos_dir> <out.json>    in vllm.sif's python; key from env VLLM_API_KEY
    python hpc/colour_check.py load <base_url> <photos_dir> <out_dir>    loop the photo set as background VLM load until <out_dir>/STOP
    python hpc/colour_check.py sim <out_dir> <seconds> <idx>             in nav116.sif: step the full house + a mast render / 2 s
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.colour_photos import COLOURS, QUESTION, SYSTEM, parse_answer, score  # noqa: E402


def vlm(base: str, photos: Path, out: Path) -> int:
    from openai import OpenAI

    doc = json.loads((photos / "truth.json").read_text())
    client = OpenAI(base_url=base, api_key=os.environ["VLLM_API_KEY"], timeout=600)
    model = client.models.list().data[0].id
    calls, answers = [], {}
    for pid in doc["photos"]:
        url = "data:image/png;base64," + base64.b64encode((photos / f"{pid}.png").read_bytes()).decode()
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                             {"type": "text", "text": QUESTION}]}]
        t0 = time.time()
        r = client.chat.completions.create(model=model, messages=msgs, temperature=0.0, max_tokens=8192)
        t1 = time.time()
        msg = r.choices[0].message
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
        answers[pid] = parse_answer(msg.content)
        rec = {"photo": pid, "truth": doc["truth"][pid], "t_start": t0, "t_end": t1, "latency_s": round(t1 - t0, 2),
               "content": msg.content, "answer": answers[pid], "reasoning_chars": len(reasoning),
               "finish_reason": r.choices[0].finish_reason, "completion_tokens": r.usage.completion_tokens}
        calls.append(rec)
        print(json.dumps(rec), flush=True)
    res = {"model": model, "selection": doc["selection"], "calls": calls, "score": score(doc["truth"], doc["selection"], answers)}
    out.write_text(json.dumps(res, indent=1))
    for c in COLOURS:
        s = res["score"][c]
        print(c, f'{s["n_pass"]}/{len(s["photos"])}', "PASS" if s["gate"] else "FAIL")
    print("G1-colour", "PASS" if res["score"]["all_pass"] else "FAIL")
    return 0 if res["score"]["all_pass"] else 1


def load(base: str, photos: Path, out: Path) -> int:
    """Background VLM load for the sims-beside measurement: same calls as `vlm`, repeated until out/STOP; unscored."""
    from openai import OpenAI

    doc = json.loads((photos / "truth.json").read_text())
    client = OpenAI(base_url=base, api_key=os.environ["VLLM_API_KEY"], timeout=600)
    model = client.models.list().data[0].id
    with open(out / "load_calls.jsonl", "w") as f:
        while not (out / "STOP").exists():
            for pid in doc["photos"]:
                url = "data:image/png;base64," + base64.b64encode((photos / f"{pid}.png").read_bytes()).decode()
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                     {"type": "text", "text": QUESTION}]}]
                t0 = time.time()
                r = client.chat.completions.create(model=model, messages=msgs, temperature=0.0, max_tokens=8192)
                f.write(json.dumps({"t_start": t0, "t_end": time.time(), "tokens": r.usage.completion_tokens}) + "\n")
                f.flush()
    return 0


def sim(out: Path, seconds: float, idx: int) -> int:
    import mujoco

    from sim.look import render_look, set_arm
    from sim.world.generate import LAYOUT, generate
    from sim.world.house import load_house
    from sim.robot import CAM_H, CAM_W

    m, d = load_house(generate(LAYOUT, 0, "none"))
    set_arm(m, d)
    r = mujoco.Renderer(m, CAM_H, CAM_W)
    steps = renders = 0
    render_s = 0.0
    t0 = last = time.time()
    while time.time() - t0 < seconds:
        for _ in range(200):
            mujoco.mj_step(m, d)
        steps += 200
        if time.time() - last >= 2.0:
            t1 = time.time()
            render_look(r, d)
            render_s += time.time() - t1
            renders += 1
            last = time.time()
    wall = time.time() - t0
    r.close()
    (out / f"sim_{idx}.json").write_text(json.dumps({
        "idx": idx, "t_start": t0, "t_end": t0 + wall, "steps": steps, "renders": renders, "wall_s": round(wall, 2),
        "steps_per_s": round(steps / wall), "steps_per_s_excl_render": round(steps / (wall - render_s)),
        "render_s_mean": round(render_s / max(renders, 1), 3)}))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit({"vlm": lambda: vlm(a[1], Path(a[2]), Path(a[3])), "load": lambda: load(a[1], Path(a[2]), Path(a[3])),
              "sim": lambda: sim(Path(a[1]), float(a[2]), int(a[3]))}[a[0]]())
