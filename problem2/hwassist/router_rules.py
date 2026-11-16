"""Deterministic intent -> tools router. Used when no LLM is reachable (offline,
no model installed) and as a safety net. Same tool names the LLM would call."""
from __future__ import annotations
import re

RULES: list[tuple[re.Pattern, list[str]]] = [(re.compile(p, re.I), t) for p, t in [
    (r"\b(upgrade|slots?|max(imum)?|expand|add|install|more|sodimm|dimm)\b.*\b(ram|memory)\b|\b(ram|memory)\b.*\b(upgrade|slots?|max(imum)?|expand|add|install)\b",
     ["get_memory_topology", "get_memory_summary"]),
    (r"\b(ram|memory)\b", ["get_memory_summary"]),
    (r"\b(cpu|processor|cores?|threads?|clock)\b", ["get_cpu_info"]),
    (r"\b(gpu|graphics?|video card|nvidia|radeon)\b", ["get_gpu_info"]),
    (r"\b(ssd|hdd|nvme|disk|drive|storage)\b.*\b(free|space|left|used|usage)\b|\b(free|space|left)\b.*\b(disk|storage|drive)\b",
     ["get_block_devices", "get_disk_usage"]),
    (r"\b(ssd|hdd|nvme|disk|drive|storage|partition)\b", ["get_block_devices"]),
    (r"\b(battery|charge|power health)\b", ["get_battery_info"]),
    (r"\b(os|kernel|distro|model|laptop|bios|vendor|system)\b", ["get_system_info"]),
    (r"\b(specs?|everything|overview|summary|hardware)\b",
     ["get_system_info", "get_cpu_info", "get_memory_summary", "get_block_devices", "get_gpu_info"]),
]]
FOLLOWUP = re.compile(r"^(and|what about|how about)\b", re.I)


def route(question: str) -> list[str]:
    """Union of ALL matching intents (multi-part questions), most specific first.
    The 'upgrade' rule subsumes the plain RAM rule so we don't double-report."""
    out: list[str] = []
    upgrade = bool(RULES[0][0].search(question))
    for i, (rx, tools) in enumerate(RULES):
        if i == 1 and upgrade:
            continue
        if rx.search(question):
            out += [t for t in tools if t not in out]
    return out
