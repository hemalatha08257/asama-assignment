"""Input guardrails for posts (indirect prompt-injection defence).

Layered: (1) normalise/cap, (2) heuristic detector, (3) pluggable ML guard
(Llama Guard / Prompt-Guard) hook. Posts are ALSO always delivered as inert
data (see delivery.build_payload), so the detector is defence in depth.
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

MAX_TITLE, MAX_CONTENT = 200, 4000

_INJECTION_PATTERNS = [
    (r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)", "override_instructions"),
    (r"disregard (all |any )?(the )?(previous|prior|above|system)", "override_instructions"),
    (r"you are now|act as (an? )?(admin|root|developer|system)|new (system )?instructions?:", "role_hijack"),
    (r"(reveal|print|output|send|leak|exfiltrate).{0,40}(api[ _-]?key|secret|token|password|credential|system prompt)", "secret_exfiltration"),
    (r"</?\s*(system|assistant|tool)\s*>|\[/?inst\]|<\|im_(start|end)\|>", "chat_template_tokens"),
    (r"(transfer|wire|pay|send)\s+(₹|rs\.?|inr|\$|usd)?\s*\d[\d,]*\s*(to|into)\b", "payment_instruction"),
    (r"https?://\S+\.(zip|exe|scr)\b", "suspicious_link"),
]
_COMPILED = [(re.compile(p, re.I | re.S), name) for p, name in _INJECTION_PATTERNS]


@dataclass
class SafetyVerdict:
    allowed: bool
    flags: list[str] = field(default_factory=list)
    title: str = ""
    content: str = ""


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # drop zero-width / control characters often used to smuggle instructions
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return text.strip()


def screen_post(title: str, content: str,
                ml_guard: Optional[Callable[[str], bool]] = None) -> SafetyVerdict:
    """ml_guard(text) -> True if text is UNSAFE (plug Llama Guard here)."""
    title, content = _clean(title)[:MAX_TITLE], _clean(content)[:MAX_CONTENT]
    blob = f"{title}\n{content}"
    flags = sorted({name for rx, name in _COMPILED if rx.search(blob)})
    if ml_guard and ml_guard(blob):
        flags.append("ml_guard_unsafe")
    return SafetyVerdict(allowed=not flags, flags=flags, title=title, content=content)
