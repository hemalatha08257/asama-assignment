"""Raw command output -> validated, normalised JSON (units in bytes)."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def to_bytes(text: str) -> int | None:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?B)\s*$", text, re.I)
    return int(float(m.group(1)) * _UNITS[m.group(2).upper()]) if m else None


def parse_lscpu(out: str) -> dict[str, Any]:
    kv = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            kv[k.strip()] = v.strip()
    sockets = int(kv.get("Socket(s)", 1) or 1)
    cores = int(kv.get("Core(s) per socket", 0) or 0) * sockets
    return {
        "model": kv.get("Model name", "unknown"), "architecture": kv.get("Architecture"),
        "physical_cores": cores or None, "logical_cpus": int(kv["CPU(s)"]) if "CPU(s)" in kv else None,
        "max_mhz": float(kv["CPU max MHz"]) if "CPU max MHz" in kv else None,
        "l3_cache": kv.get("L3 cache"), "virtualization": kv.get("Virtualization"),
    }


def parse_free(out: str) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for line in out.splitlines():
        p = line.split()
        if p and p[0] == "Mem:":
            res["total_bytes"], res["used_bytes"] = int(p[1]), int(p[2])
            res["available_bytes"] = int(p[6]) if len(p) > 6 else int(p[3])
        elif p and p[0] == "Swap:":
            res["swap_total_bytes"] = int(p[1])
    return res


def parse_dmidecode_memory(out: str) -> dict[str, Any]:
    """Block-aware (the naive 'Size:' regex also matches Cache/Volatile Size lines)."""
    blocks = re.split(r"\n\s*\n", out)
    max_cap, array_slots, modules = None, None, []
    for b in blocks:
        if re.search(r"^Physical Memory Array\s*$", b, re.M):
            m = re.search(r"^\s*Maximum Capacity:\s*(.+)$", b, re.M)
            n = re.search(r"^\s*Number Of Devices:\s*(\d+)", b, re.M)
            max_cap = to_bytes(m.group(1)) if m else max_cap
            array_slots = int(n.group(1)) if n else array_slots
        elif re.search(r"^Memory Device\s*$", b, re.M):
            g = lambda key: (re.search(rf"^\s*{key}:\s*(.+)$", b, re.M) or [None, None])[1]
            size = g("Size")
            modules.append({
                "locator": g("Locator"), "installed": bool(size) and "No Module" not in size,
                "size_bytes": to_bytes(size) if size else None, "type": g("Type"),
                "speed": g("Configured Memory Speed") or g("Speed"), "form_factor": g("Form Factor"),
                "manufacturer": g("Manufacturer"),
            })
    installed = [m for m in modules if m["installed"]]
    total = len(modules) or array_slots or 0
    inst_bytes = sum(m["size_bytes"] or 0 for m in installed)
    out_d = {"source": "dmidecode", "maximum_capacity_bytes": max_cap, "total_slots": total,
             "used_slots": len(installed), "free_slots": total - len(installed),
             "installed_bytes": inst_bytes, "modules": installed}
    if max_cap:
        out_d["upgrade_headroom_bytes"] = max(0, max_cap - inst_bytes)
    return out_d


def parse_lsblk(out: str) -> dict[str, Any]:
    devs = json.loads(out).get("blockdevices", [])
    disks = []
    for d in devs:
        if d.get("type") != "disk" or int(d.get("size") or 0) == 0 or re.match(r"^(zram|loop|ram)", d.get("name", "")):
            continue
        rota = d.get("rota")
        kind = "HDD" if rota in (True, 1, "1") else "SSD/NVMe"
        if (d.get("tran") or "") == "nvme" or str(d.get("name", "")).startswith("nvme"):
            kind = "NVMe SSD"
        disks.append({"name": d["name"], "size_bytes": int(d["size"]), "model": (d.get("model") or "").strip(),
                      "kind": kind, "interface": d.get("tran"),
                      "partitions": [{"name": c["name"], "size_bytes": int(c["size"]), "mount": c.get("mountpoint")}
                                     for c in d.get("children", [])]})
    return {"disks": disks}


def parse_lspci(out: str) -> dict[str, Any]:
    gpus = []
    for line in out.splitlines():
        parts = re.findall(r'"([^"]*)"', line)
        if len(parts) >= 3 and any(k in parts[0] for k in ("VGA", "3D controller", "Display controller")):
            vendor, dev = parts[1], parts[2]
            gpus.append({"vendor": vendor, "device": dev,
                         "type": "integrated" if re.search(r"Intel|Radeon (Vega|Graphics)|AMD/ATI.*(Renoir|Cezanne|Phoenix)", vendor + dev) else "discrete/unknown"})
    return {"gpus": gpus}


def parse_uname(out: str) -> dict[str, Any]:
    p = out.split()
    return {"kernel": " ".join(p[:2]) if len(p) >= 2 else out.strip(), "machine": p[2] if len(p) > 2 else None}


def parse_df(out: str) -> dict[str, Any]:
    rows = []
    for line in out.splitlines()[1:]:
        p = line.split()
        if len(p) >= 6 and p[0].startswith("/dev/"):
            rows.append({"device": p[0], "size_bytes": int(p[1]), "used_bytes": int(p[2]),
                         "available_bytes": int(p[3]), "mount": p[5]})
    return {"filesystems": rows}


def read_battery(sysfs: Path) -> dict[str, Any]:
    base = sysfs / "class" / "power_supply"
    if not base.exists():
        return {"present": False}
    for d in sorted(base.glob("BAT*")):
        rd = lambda n: (d / n).read_text().strip() if (d / n).exists() else None
        full, design = rd("energy_full") or rd("charge_full"), rd("energy_full_design") or rd("charge_full_design")
        res = {"present": True, "name": d.name, "status": rd("status"),
               "percent": int(rd("capacity")) if rd("capacity") else None}
        if full and design and int(design):
            res["health_percent"] = round(100 * int(full) / int(design), 1)
        return res
    return {"present": False}


PARSERS = {"get_cpu_info": parse_lscpu, "get_memory_summary": parse_free,
           "get_memory_topology": parse_dmidecode_memory, "get_block_devices": parse_lsblk,
           "get_gpu_info": parse_lspci, "get_system_info": parse_uname, "get_disk_usage": parse_df}


def parse(name: str, out: str) -> dict[str, Any]:
    try:
        return PARSERS[name](out)
    except Exception as e:  # never feed the LLM a half-parsed blob
        return {"error": "parse_failed", "tool": name, "reason": type(e).__name__}
