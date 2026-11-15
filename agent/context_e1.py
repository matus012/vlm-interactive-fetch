"""E1 rules text (task 009, context row 22): the frozen rules text minus rules line E (task 006), nothing else.

    SYSTEM_NO_E: agent.context.SYSTEM with RULE_E removed (it occurs exactly once)
    build_packet_no_e(...): agent.context.build_packet with that rules text

Selected only by `sim.agent_run --e1`; agent/context.py (an interface-freeze file) is untouched.
"""
from __future__ import annotations

from dataclasses import replace

from agent.context import SYSTEM, Packet, build_packet

RULE_E = ('A door blocked by something push reports "failed: not movable" stays blocked; to go round it, name the '
          'other door in via. ')
assert SYSTEM.count(RULE_E) == 1, "rules line E not found once in the frozen rules text"
SYSTEM_NO_E = SYSTEM.replace(RULE_E, "")


def build_packet_no_e(*args, **kwargs) -> Packet:  # noqa: ANN002, ANN003 - same signature as build_packet
    return replace(build_packet(*args, **kwargs), system=SYSTEM_NO_E)
