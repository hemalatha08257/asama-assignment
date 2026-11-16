"""In-memory repository. `schema.sql` + README describe the Postgres/pgvector
equivalent; the router only depends on this small interface."""
from __future__ import annotations
from collections import defaultdict
from .models import Agent, Post, Comment, RoutingLog


class MemoryStore:
    def __init__(self) -> None:
        self.agents: dict[str, Agent] = {}
        self.posts: dict[str, Post] = {}
        self.comments: dict[str, list[Comment]] = defaultdict(list)
        self.logs: list[RoutingLog] = []

    def add_agent(self, a: Agent) -> Agent:
        self.agents[a.id] = a
        return a

    def add_post(self, p: Post) -> Post:
        self.posts[p.id] = p
        return p

    def add_comment(self, c: Comment) -> Comment:
        self.comments[c.post_id].append(c)
        return c

    def add_log(self, l: RoutingLog) -> RoutingLog:
        self.logs.append(l)
        return l

    def log_for(self, post_id: str, agent_id: str) -> RoutingLog | None:
        for l in self.logs:
            if l.post_id == post_id and l.target_agent_id == agent_id:
                return l
        return None

    def logs_for_agent(self, agent_id: str) -> list[RoutingLog]:
        return [l for l in self.logs if l.target_agent_id == agent_id]
