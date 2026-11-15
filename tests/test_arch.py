"""G2-arch: the agent side imports no MuJoCo / numpy / sim.* (static, ast). The only allowed sim module is
sim.constants (stdlib-only; asserted below). No other exceptions. Scope: agent/*.py except the adapter-facing
modules that do not exist on the agent side; every module under agent/ and everything it imports from agent/."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED = {"mujoco", "numpy", "scipy", "cv2", "torch", "PIL", "yaml"}
ALLOWED_SIM = {"sim.constants"}


def imports(path: Path) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


def closure() -> dict[Path, set[str]]:
    todo, seen = sorted((ROOT / "agent").glob("*.py")), {}
    while todo:
        p = todo.pop()
        if p in seen or not p.exists():
            continue
        seen[p] = imports(p)
        for m in seen[p]:
            f = ROOT / (m.replace(".", "/") + ".py")
            if m.split(".")[0] in ("agent", "sim") and f.exists():
                todo.append(f)
    return seen


def test_agent_side_has_no_mujoco_numpy_sim():
    bad = []
    for p, mods in closure().items():
        for m in mods:
            top = m.split(".")[0]
            if top in BANNED or (top == "sim" and not any(m == a or m.startswith(a + ".") or a.startswith(m + ".") for a in ALLOWED_SIM)):
                bad.append((str(p.relative_to(ROOT)), m))
    assert not bad, bad


def test_sim_constants_is_stdlib_only():
    assert not any(m.split(".")[0] in BANNED | {"sim"} for m in imports(ROOT / "sim" / "constants.py"))


def test_closure_covers_the_agent_modules():
    names = {p.name for p in closure()}
    assert {"skills.py", "runtime.py", "scripts.py", "fake_robot.py", "tools.py", "validate.py", "context.py", "leak.py",
            "vlm.py", "loop.py"} <= names
