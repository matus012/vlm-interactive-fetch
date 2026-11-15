"""The six-skill tool list (OpenAI function-tool JSON schemas). Args are names, never coordinates. Stdlib only."""
from __future__ import annotations


def _tool(name: str, desc: str, props: dict[str, str], required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS: list[dict] = [
    _tool("navigate", "Drive to a named place. Returns 'reached' or 'failed: blocked', plus a new photo.",
          {"place": "Name of a place, e.g. 'kitchen table'.",
           "via": "Optional room or door name to route through (a detour)."}, ["place"]),
    _tool("look", "Turn in place to face four fixed directions and take one photo in each. Returns four numbered "
          "photos.", {}, []),
    _tool("locate", "Find the named object in the current photos.",
          {"object": "Object name, e.g. 'red box'."}, ["object"]),
    _tool("pick", "Pick up a previously located box.", {"object": "Object name, e.g. 'red box'."}, ["object"]),
    _tool("place", "Put the held box down on a named surface.", {"surface": "Surface name, e.g. 'kitchen table'."},
          ["surface"]),
    _tool("push", "Push a blocking object out of the way with the base. Returns 'pushed', 'failed: not movable' (it "
          "did not move at all), 'failed: stuck' (it moved but could not be cleared) or 'failed: not blocking' (it "
          "blocks no doorway and no place, so nothing was pushed).",
          {"object": "Object name, e.g. 'crate'."},
          ["object"]),
    _tool("done", "Stop the run. The note is for the demo only.", {"note": "Short free-text note."}, ["note"]),
]

SKILLS: dict[str, dict] = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
