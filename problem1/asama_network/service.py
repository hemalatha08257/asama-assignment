"""Orchestrates: register -> publish (screen, embed, route, deliver) -> comment."""
from __future__ import annotations
from .delivery import Deliverer
from .embeddings import Embedder, ConceptHashEmbedder
from .feedback import FeedbackLearner
from .models import Agent, Post, Comment, RoutingLog, default_expiry
from .reranker import Reranker, ConceptReranker
from .router import Router, RouterConfig
from .safety import screen_post
from .store import MemoryStore


class NetworkService:
    def __init__(self, store: MemoryStore | None = None, embedder: Embedder | None = None,
                 reranker: Reranker | None = None, deliverer: Deliverer | None = None,
                 cfg: RouterConfig | None = None, ml_guard=None):
        self.store = store or MemoryStore()
        self.embedder = embedder or ConceptHashEmbedder()
        self.router = Router(self.store, reranker or ConceptReranker(), cfg)
        self.deliverer = deliverer or Deliverer()
        self.learner = FeedbackLearner(self.store)
        self.ml_guard = ml_guard

    def register_agent(self, **kw) -> Agent:
        a = Agent(**kw)
        a.interest_embedding = self.embedder.embed(a.interest_manifesto)
        return self.store.add_agent(a)

    def publish_post(self, author_agent_id: str, title: str, content: str) -> dict:
        if author_agent_id not in self.store.agents:
            raise KeyError("unknown author agent")
        verdict = screen_post(title, content, self.ml_guard)
        if not verdict.allowed:
            return {"status": "REJECTED", "flags": verdict.flags}
        post = Post(author_agent_id, verdict.title, verdict.content, expires_at=default_expiry())
        post.post_embedding = self.embedder.embed(post.text)
        self.store.add_post(post)
        author = self.store.agents[author_agent_id]
        notified = []
        for c in self.router.route(post):
            ok = self.deliverer.deliver(post, author, c.agent, c.rerank_score)
            self.store.add_log(RoutingLog(post.id, c.agent.id, round(c.vector_score, 4),
                                          c.rerank_score, "NOTIFIED" if ok else "FAILED"))
            if ok:
                notified.append({"agent": c.agent.company_name, "score": c.rerank_score})
        return {"status": "PUBLISHED", "post_id": post.id, "notified": notified}

    def add_comment(self, post_id: str, author_agent_id: str, content: str) -> Comment:
        post, agent = self.store.posts[post_id], self.store.agents[author_agent_id]
        c = self.store.add_comment(Comment(post_id, author_agent_id, screen_post("", content).content))
        if self.store.log_for(post_id, author_agent_id):
            self.learner.on_engaged(agent, post)
        return c
