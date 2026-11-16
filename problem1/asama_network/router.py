"""3-stage routing: SQL-style guardrails -> ANN top-K -> rerank + threshold."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from .embeddings import cosine, normalize
from .models import Agent, Post
from .reranker import Reranker
from .store import MemoryStore


@dataclass
class Candidate:
    agent: Agent
    vector_score: float
    rerank_score: float = 0.0


@dataclass
class RouterConfig:
    top_k: int = 50
    threshold: float = 0.75
    max_notify: int = 5
    min_trust: float = 0.2
    dynamic_blend: float = 0.35  # weight of feedback-learned vector


class Router:
    def __init__(self, store: MemoryStore, reranker: Reranker, cfg: RouterConfig | None = None):
        self.store, self.reranker, self.cfg = store, reranker, cfg or RouterConfig()

    # Stage 1 -- hard metadata filters (SQL WHERE in production)
    def stage1_filter(self, post: Post) -> list[Agent]:
        author = self.store.agents[post.author_agent_id]
        now = datetime.now(timezone.utc)
        if post.expires_at and post.expires_at < now:
            return []
        return [
            a for a in self.store.agents.values()
            if a.active and a.id != author.id
            and a.trust_score >= self.cfg.min_trust
            and author.industry_category not in a.blocked_industries
            and a.industry_category not in author.blocked_industries
        ]

    def _agent_vector(self, a: Agent) -> list[float]:
        if a.dynamic_embedding is None:
            return a.interest_embedding
        w = self.cfg.dynamic_blend
        return normalize([(1 - w) * s + w * d for s, d in zip(a.interest_embedding, a.dynamic_embedding)])

    # Stage 2 -- ANN (brute-force here; pgvector HNSW `<=>` in production)
    def stage2_ann(self, post: Post, agents: list[Agent]) -> list[Candidate]:
        scored = [Candidate(a, cosine(post.post_embedding, self._agent_vector(a))) for a in agents]
        scored.sort(key=lambda c: c.vector_score, reverse=True)
        return scored[: self.cfg.top_k]

    # Stage 3 -- joint rerank + threshold, trust-weighted
    def stage3_rerank(self, post: Post, cands: list[Candidate]) -> list[Candidate]:
        for c in cands:
            c.rerank_score = round(self.reranker.score(post, c.agent, c.vector_score)
                                   * min(1.0, 0.5 + 0.5 * c.agent.trust_score), 4)
        keep = [c for c in cands if c.rerank_score >= self.cfg.threshold]
        keep.sort(key=lambda c: c.rerank_score, reverse=True)
        return keep[: self.cfg.max_notify]

    def route(self, post: Post) -> list[Candidate]:
        return self.stage3_rerank(post, self.stage2_ann(post, self.stage1_filter(post)))
