"""Closed-loop learning: engagement shifts each agent's dynamic interest vector."""
from __future__ import annotations
from .embeddings import normalize
from .models import Agent, Post
from .store import MemoryStore


class FeedbackLearner:
    def __init__(self, store: MemoryStore, lr_pos: float = 0.15, lr_neg: float = 0.10,
                 ignore_streak: int = 3):
        self.store, self.lr_pos, self.lr_neg, self.ignore_streak = store, lr_pos, lr_neg, ignore_streak

    def engagement_rate(self, agent_id: str) -> float:
        logs = self.store.logs_for_agent(agent_id)
        sent = [l for l in logs if l.status in ("NOTIFIED", "ENGAGED", "IGNORED")]
        return sum(1 for l in sent if l.status == "ENGAGED") / len(sent) if sent else 0.0

    def _shift(self, agent: Agent, post: Post, delta: float) -> None:
        base = agent.dynamic_embedding or agent.interest_embedding
        agent.dynamic_embedding = normalize([b + delta * p for b, p in zip(base, post.post_embedding)])

    def on_engaged(self, agent: Agent, post: Post) -> None:
        log = self.store.log_for(post.id, agent.id)
        if log:
            log.status = "ENGAGED"
        self._shift(agent, post, +self.lr_pos)
        agent.trust_score = min(1.0, agent.trust_score + 0.02)

    def on_ignored(self, agent: Agent, post: Post) -> None:
        """Called by a sweeper once a notification passes its response window."""
        log = self.store.log_for(post.id, agent.id)
        if log and log.status == "NOTIFIED":
            log.status = "IGNORED"
        recent = self.store.logs_for_agent(agent.id)[-self.ignore_streak:]
        if len(recent) == self.ignore_streak and all(l.status == "IGNORED" for l in recent):
            self._shift(agent, post, -self.lr_neg)  # push away from unengaged topic
