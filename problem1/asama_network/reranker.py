"""Stage-3 precision scoring.

`ConceptReranker` is an offline stand-in for a cross-encoder (bge-reranker /
Cohere Rerank). It scores the *pair* (post, agent interests) jointly:
  - dense similarity (semantic proximity)
  - concept coverage: how many of the post's business concepts the agent cares about
  - complementarity bonus: post concept pairs an agent could *act on*
    (e.g. food-delivery promo  x  payments agent => partnership opportunity)
"""
from __future__ import annotations
from typing import Protocol
from .embeddings import concepts_in
from .models import Agent, Post

# (post concept, agent concept) pairs that signal a partnership opportunity
COMPLEMENTS = {
    ("food", "payments"), ("ecommerce", "payments"), ("promo", "payments"),
    ("auto", "lending"), ("ecommerce", "logistics"), ("food", "logistics"),
    ("festive", "promo"),
}


class Reranker(Protocol):
    def score(self, post: Post, agent: Agent, dense: float) -> float: ...


class ConceptReranker:
    def score(self, post: Post, agent: Agent, dense: float) -> float:
        pc, ac = concepts_in(post.text), concepts_in(agent.interest_manifesto)
        if not pc:
            return max(0.0, dense) * 0.5
        coverage = len(pc & ac) / len(pc)
        comp = sum(1 for (p, a) in COMPLEMENTS if p in pc and a in ac)
        comp_bonus = min(0.3, 0.15 * comp)
        score = 0.45 * max(0.0, dense) + 0.40 * coverage + comp_bonus
        return round(min(1.0, score), 4)


class CohereReranker:  # pragma: no cover - needs network + key
    def __init__(self, model: str = "rerank-v3.5"):
        import cohere
        self.co, self.model = cohere.Client(), model

    def score(self, post, agent, dense):
        r = self.co.rerank(model=self.model, query=post.text,
                           documents=[agent.interest_manifesto], top_n=1)
        return r.results[0].relevance_score
