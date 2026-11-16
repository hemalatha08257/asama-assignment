"""Optional local-LLM backend (Ollama). Offline default works without it."""
from __future__ import annotations
import json
import urllib.request


class OllamaClient:  # pragma: no cover - needs a running ollama
    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://127.0.0.1:11434"):
        self.model, self.host = model, host

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": 0}}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["message"]

    def alive(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=1)
            return True
        except Exception:
            return False
