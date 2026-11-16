"""Hardened command execution: no shell, absolute paths, timeout, output cap,
sanitised environment, scoped sudo for the single root-only tool."""
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from . import parsers
from .registry import TOOLS, discover

MAX_OUTPUT = 512 * 1024
SAFE_ENV = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/sbin:/usr/bin:/sbin:/bin"}

Runner = Callable[[list[str]], tuple[int, str, str]]


def default_runner(argv: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(argv, capture_output=True, text=True, timeout=5, shell=False,
                       env=SAFE_ENV, stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout[:MAX_OUTPUT], p.stderr[:2000]


class Executor:
    def __init__(self, runner: Runner = default_runner, sysfs: Path = Path("/sys"),
                 available: dict[str, str] | None = None, use_sudo: bool = True):
        self.run, self.sysfs, self.use_sudo = runner, sysfs, use_sudo
        self.available = discover() if available is None else available

    def call(self, name: str, args: dict | None = None) -> dict[str, Any]:
        spec = TOOLS.get(name)
        if spec is None:                                      # allow-list check
            return {"error": "unknown_tool", "tool": name}
        if name not in self.available:
            return self._fallback(name, "binary not installed")
        if args:                                              # none of our tools accept free-form args
            return {"error": "unexpected_arguments", "tool": name}
        if name == "get_battery_info":
            return parsers.read_battery(self.sysfs)
        argv = [self.available[name], *spec.argv]
        if spec.needs_root and os.geteuid() != 0 and self.use_sudo:
            argv = ["sudo", "-n", *argv]                      # -n: never prompt, fail fast
        try:
            rc, out, err = self.run(argv)
        except Exception as e:                                # timeout, OSError...
            return self._fallback(name, f"{type(e).__name__}")
        if rc != 0:
            return self._fallback(name, err.strip().splitlines()[-1] if err.strip() else f"exit {rc}")
        return parsers.parse(name, out)

    def _fallback(self, name: str, reason: str) -> dict[str, Any]:
        if name == "get_memory_topology":
            data = self.call("get_memory_summary")
            data.update({"source": "fallback:/proc/meminfo",
                         "limitation": "Slot count and motherboard max RAM need root access to DMI data "
                                       "(see sudoers setup in README). Reason: " + reason})
            return data
        return {"error": "execution_failed", "tool": name, "reason": reason}
