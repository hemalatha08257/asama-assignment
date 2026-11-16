from __future__ import annotations
import json
from typing import Any

from . import answer as ans
from .executor import Executor
from .registry import TOOLS, tool_schemas
from .router_rules import route, FOLLOWUP


class Assistant:
    """Session object: caches tool results (multi-turn follow-ups) and runs either
    the LLM tool-calling loop (if a client is given) or the deterministic router."""

    def __init__(self, executor: Executor | None = None, llm=None, max_tool_rounds: int = 4):
        self.ex = executor or Executor()
        self.llm, self.max_rounds = llm, max_tool_rounds
        self.cache: dict[str, dict] = {}
        self.history: list[dict] = []

    def _tool(self, name: str, args: dict | None = None) -> dict:
        if name not in self.cache or "error" in self.cache[name]:
            self.cache[name] = self.ex.call(name, args)
        return self.cache[name]

    def ask(self, question: str) -> str:
        reply = self._ask_llm(question) if self.llm else self._ask_rules(question)
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
        return reply

    # ---- offline / deterministic ----
    def _ask_rules(self, q: str) -> str:
        tools = route(q)
        if not tools and FOLLOWUP.match(q.strip()) and self.history:
            tools = route(q)  # keyword in follow-up ("and the GPU?") still routes; else fall through
        data = {t: self._tool(t) for t in tools}
        return ans.compose(q, tools, data)

    # ---- LLM tool-calling loop ----
    def _ask_llm(self, q: str) -> str:
        msgs = [{"role": "system", "content": ans.GROUNDING_PROMPT +
                 " Already-gathered data this session: " + json.dumps(self.cache)[:6000]}] + self.history[-8:]
        msgs.append({"role": "user", "content": q})
        schemas = tool_schemas(self.ex.available)
        for _ in range(self.max_rounds):
            m = self.llm.chat(msgs, schemas)
            calls = m.get("tool_calls") or []
            if not calls:
                return m.get("content", "").strip() or ans.UNKNOWN
            msgs.append(m)
            for c in calls:
                fn = c["function"]
                name, args = fn["name"], fn.get("arguments") or {}
                result = self._tool(name, args) if name in TOOLS else {"error": "unknown_tool"}
                msgs.append({"role": "tool", "name": name, "content": json.dumps(result)})
        return "I could not complete that within the allowed number of tool calls."
