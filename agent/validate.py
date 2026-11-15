"""Check one tool call against the tool list. Stdlib only (runs inside any python, incl. vllm.sif)."""
from __future__ import annotations

import json

from agent.tools import SKILLS


def validate_call(name: str, arguments: str | dict | None) -> tuple[bool, str]:
    """(ok, reason). `arguments` is the JSON string a tool call carries (or an already parsed dict)."""
    if name not in SKILLS:
        return False, f"unknown skill {name!r}; valid: {', '.join(SKILLS)}"
    if arguments is None or arguments == "":
        arguments = "{}"
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as e:
            return False, f"arguments are not valid JSON: {e}"
    else:
        args = arguments
    if not isinstance(args, dict):
        return False, "arguments must be a JSON object"
    schema = SKILLS[name]
    props, required = schema["properties"], schema["required"]
    for k in args:
        if k not in props:
            return False, f"{name}: unexpected argument {k!r}"
    for k in required:
        if k not in args:
            return False, f"{name}: missing required argument {k!r}"
    for k, v in args.items():
        if not isinstance(v, str):
            return False, f"{name}: argument {k!r} must be a string, got {type(v).__name__}"
        if not v.strip():
            return False, f"{name}: argument {k!r} is empty"
    return True, "ok"
