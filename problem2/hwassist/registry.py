"""Allow-list of tools. The LLM only ever picks a tool NAME + validated params.
Every argv is fixed here; user/LLM text never reaches a command line unvalidated."""
from __future__ import annotations
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable

_DEVICE_RE = re.compile(r"^[a-zA-Z0-9]{1,16}$")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    binary: str                        # resolved via shutil.which at discovery time
    argv: tuple[str, ...]              # fixed arguments
    needs_root: bool = False
    params: dict[str, Any] = field(default_factory=dict)  # JSON-schema properties
    fallback: str | None = None        # name of non-root fallback reader


TOOLS: dict[str, ToolSpec] = {t.name: t for t in [
    ToolSpec("get_cpu_info", "CPU model, cores, threads, frequency, cache, architecture.", "lscpu", ()),
    ToolSpec("get_memory_summary", "Installed RAM, used/free/available and swap (bytes).", "free", ("-b",)),
    ToolSpec("get_memory_topology",
             "Physical RAM modules, total/used/free slots, module speed/type and the motherboard's "
             "maximum supported capacity. Use for upgrade questions.",
             "dmidecode", ("-t", "memory"), needs_root=True, fallback="memory_summary_only"),
    ToolSpec("get_block_devices", "Disks/partitions: size, type, model, SSD/HDD, interface, mountpoints.",
             "lsblk", ("-J", "-b", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,MODEL,ROTA,TRAN")),
    ToolSpec("get_gpu_info", "Graphics adapters (integrated and discrete).", "lspci", ("-mm",)),
    ToolSpec("get_system_info", "Laptop vendor/model, BIOS version, OS release and kernel.", "uname", ("-srm",)),
    ToolSpec("get_battery_info", "Battery presence, charge, design capacity and health.", "", ()),  # reads /sys
    ToolSpec("get_disk_usage", "Free/used space for a mounted filesystem.", "df", ("-B1",)),
]}

# Parameterised tools: map validated params -> extra fixed-shape argv
PARAM_VALIDATORS: dict[str, dict[str, Callable[[Any], str]]] = {
    "get_block_devices": {},
    "get_disk_usage": {},
}


def validate_device(value: Any) -> str:
    if not isinstance(value, str) or not _DEVICE_RE.match(value):
        raise ValueError("invalid device name")
    return value


def discover(which: Callable[[str], str | None] = shutil.which) -> dict[str, str]:
    """Return {tool_name: absolute_binary_path} for tools actually present."""
    found = {}
    for name, spec in TOOLS.items():
        if spec.binary == "":
            found[name] = ""  # pseudo-tool backed by /sys readers
        else:
            p = which(spec.binary) or (which("/usr/sbin/" + spec.binary))
            if p:
                found[name] = p
    return found


def tool_schemas(available: dict[str, str]) -> list[dict]:
    """OpenAI/Ollama-style function schemas -- ONLY for tools that exist here."""
    return [{"type": "function", "function": {
        "name": n, "description": TOOLS[n].description,
        "parameters": {"type": "object", "properties": TOOLS[n].params, "required": []}}}
        for n in available]
