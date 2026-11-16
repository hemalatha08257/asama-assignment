import json, tempfile, unittest
from pathlib import Path

from hwassist import parsers
from hwassist.answer import compose
from hwassist.assistant import Assistant
from hwassist.executor import Executor
from hwassist.registry import TOOLS, discover, tool_schemas, validate_device
from hwassist.router_rules import route

DMI = """# dmidecode 3.5
Handle 0x0010, DMI type 16, 23 bytes
Physical Memory Array
\tLocation: System Board Or Motherboard
\tUse: System Memory
\tError Correction Type: None
\tMaximum Capacity: 64 GB
\tNumber Of Devices: 2

Handle 0x0011, DMI type 17, 40 bytes
Memory Device
\tSize: 16 GB
\tForm Factor: SODIMM
\tLocator: DIMM A
\tType: DDR4
\tSpeed: 3200 MT/s
\tConfigured Memory Speed: 3200 MT/s
\tManufacturer: Samsung
\tCache Size: None
\tLogical Size: None

Handle 0x0012, DMI type 17, 40 bytes
Memory Device
\tSize: No Module Installed
\tForm Factor: Unknown
\tLocator: DIMM B
\tType: Unknown
\tSpeed: Unknown
"""
FREE = "              total        used        free      shared  buff/cache   available\nMem:    16637468672  5000000000  2000000000   100000000  9000000000  11000000000\nSwap:    2147479552           0  2147479552\n"
LSBLK = json.dumps({"blockdevices": [
    {"name": "nvme0n1", "size": "512110190592", "type": "disk", "model": "Samsung SSD 980", "rota": False, "tran": "nvme",
     "children": [{"name": "nvme0n1p1", "size": "536870912", "type": "part", "mountpoint": "/boot/efi"}]},
    {"name": "loop0", "size": "0", "type": "loop"}]})
LSPCI = '00:02.0 "VGA compatible controller" "Intel Corporation" "Alder Lake-P GT2 [Iris Xe Graphics]" -r0c\n01:00.0 "3D controller" "NVIDIA Corporation" "GA107M [RTX 3050 Mobile]" -ra1\n'
LSCPU = "Architecture: x86_64\nModel name: 12th Gen Intel(R) Core(TM) i7-1255U\nCPU(s): 12\nCore(s) per socket: 10\nSocket(s): 1\nCPU max MHz: 4700.0000\n"


def fake_runner(table, calls=None):
    def run(argv):
        if calls is not None: calls.append(argv)
        key = argv[-1] if argv[0] != "sudo" else argv[-2] + " " + argv[-1]
        for k, v in table.items():
            if k in " ".join(argv): return v
        return 1, "", "not found"
    return run


def make(table, avail=None, **kw):
    avail = avail or {n: "/usr/bin/" + (t.binary or "x") for n, t in TOOLS.items()}
    return Executor(runner=fake_runner(table), available=avail, use_sudo=False, **kw)


class ParserTests(unittest.TestCase):
    def test_dmidecode_blocks_not_fooled_by_cache_size(self):
        d = parsers.parse_dmidecode_memory(DMI)
        self.assertEqual((d["total_slots"], d["used_slots"], d["free_slots"]), (2, 1, 1))
        self.assertEqual(d["installed_bytes"], 16 * 1024**3)
        self.assertEqual(d["upgrade_headroom_bytes"], 48 * 1024**3)
        self.assertEqual(d["modules"][0]["type"], "DDR4")

    def test_free_lsblk_lspci_lscpu(self):
        self.assertEqual(parsers.parse_free(FREE)["total_bytes"], 16637468672)
        disks = parsers.parse_lsblk(LSBLK)["disks"]
        self.assertEqual(len(disks), 1); self.assertEqual(disks[0]["kind"], "NVMe SSD")
        g = parsers.parse_lspci(LSPCI)["gpus"]
        self.assertEqual(len(g), 2); self.assertEqual(g[0]["type"], "integrated")
        c = parsers.parse_lscpu(LSCPU)
        self.assertEqual((c["physical_cores"], c["logical_cpus"]), (10, 12))

    def test_garbage_input_yields_error_not_exception(self):
        self.assertEqual(parsers.parse("get_block_devices", "not json")["error"], "parse_failed")


class SecurityTests(unittest.TestCase):
    def test_unknown_tool_and_args_rejected(self):
        ex = make({})
        self.assertEqual(ex.call("rm_rf")["error"], "unknown_tool")
        self.assertEqual(ex.call("get_cpu_info", {"flags": "; rm -rf /"})["error"], "unexpected_arguments")

    def test_no_shell_and_fixed_argv(self):
        calls = []
        ex = Executor(runner=fake_runner({"lscpu": (0, LSCPU, "")}, calls),
                      available={"get_cpu_info": "/usr/bin/lscpu"}, use_sudo=False)
        ex.call("get_cpu_info")
        self.assertEqual(calls, [["/usr/bin/lscpu"]])

    def test_dmidecode_gets_sudo_non_interactive_when_not_root(self):
        import os; 
        if os.geteuid() == 0: self.skipTest("running as root")
        calls = []
        ex = Executor(runner=fake_runner({"dmidecode": (0, DMI, "")}, calls),
                      available={"get_memory_topology": "/usr/sbin/dmidecode"}, use_sudo=True)
        ex.call("get_memory_topology")
        self.assertEqual(calls[0], ["sudo", "-n", "/usr/sbin/dmidecode", "-t", "memory"])

    def test_device_validator(self):
        self.assertEqual(validate_device("nvme0n1"), "nvme0n1")
        for bad in ("sda; reboot", "../etc", "", None, "a" * 40):
            with self.assertRaises(ValueError): validate_device(bad)

    def test_schemas_only_for_available_tools(self):
        names = [s["function"]["name"] for s in tool_schemas({"get_cpu_info": "/usr/bin/lscpu"})]
        self.assertEqual(names, ["get_cpu_info"])
        self.assertNotIn("get_memory_topology", discover(lambda b: "/usr/bin/lscpu" if b == "lscpu" else None))


class AssistantTests(unittest.TestCase):
    def table(self):
        return {"dmidecode": (0, DMI, ""), "free": (0, FREE, ""), "lscpu": (0, LSCPU, ""),
                "lspci": (0, LSPCI, ""), "lsblk": (0, LSBLK, "")}

    def test_upgrade_question(self):
        bot = Assistant(make(self.table()))
        a = bot.ask("Can I upgrade my RAM, and by how much?")
        self.assertIn("16 GB in 1 of 2", a); self.assertIn("64 GB", a); self.assertIn("48 GB more", a)

    def test_missing_privilege_degrades_honestly(self):
        t = self.table(); t["dmidecode"] = (1, "", "sudo: a password is required")
        a = Assistant(make(t)).ask("how much RAM can I add?")
        self.assertIn("cannot determine", a); self.assertNotIn("64 GB", a)

    def test_multi_intent_and_followup_cache(self):
        calls = []
        ex = Executor(runner=fake_runner(self.table(), calls),
                      available={n: "/usr/bin/" + (t.binary or "x") for n, t in TOOLS.items()}, use_sudo=False)
        bot = Assistant(ex)
        a = bot.ask("what cpu do I have and what gpu?")
        self.assertIn("i7-1255U", a); self.assertIn("RTX 3050", a)
        n = len(calls); bot.ask("and the CPU again?")
        self.assertEqual(len(calls), n)  # served from session cache, no re-exec

    def test_routing_and_out_of_scope(self):
        self.assertEqual(route("can I upgrade my ram")[0], "get_memory_topology")
        self.assertEqual(route("how much ram do i have"), ["get_memory_summary"])
        self.assertEqual(route("tell me a joke"), [])
        self.assertIn("I can answer questions", compose("joke", [], {}))

    def test_battery_from_sysfs(self):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d) / "class" / "power_supply" / "BAT0"; b.mkdir(parents=True)
            for k, v in dict(capacity="80", status="Discharging", energy_full="40000000", energy_full_design="50000000").items():
                (b / k).write_text(v)
            r = parsers.read_battery(Path(d))
            self.assertEqual((r["percent"], r["health_percent"]), (80, 80.0))


class LLMLoopTests(unittest.TestCase):
    def test_llm_tool_call_loop_and_unknown_tool_blocked(self):
        class FakeLLM:
            def __init__(self): self.n = 0
            def chat(self, msgs, tools):
                self.n += 1
                if self.n == 1:
                    return {"role": "assistant", "content": "", "tool_calls": [
                        {"function": {"name": "get_cpu_info", "arguments": {}}},
                        {"function": {"name": "bash", "arguments": {"cmd": "rm -rf /"}}}]}
                tool_msgs = [m for m in msgs if m["role"] == "tool"]
                assert any(json.loads(m["content"]).get("error") == "unknown_tool" for m in tool_msgs)
                return {"role": "assistant", "content": "You have an i7-1255U."}
        bot = Assistant(make({"lscpu": (0, LSCPU, "")}), llm=FakeLLM())
        self.assertIn("i7-1255U", bot.ask("what cpu?"))


if __name__ == "__main__":
    unittest.main()
