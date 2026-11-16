from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Agent:
    company_name: str
    industry_category: str
    webhook_url: str
    interest_manifesto: str
    webhook_secret: str = field(default_factory=lambda: uuid.uuid4().hex)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active: bool = True
    trust_score: float = 1.0
    blocked_industries: set[str] = field(default_factory=set)
    interest_embedding: list[float] = field(default_factory=list)
    dynamic_embedding: Optional[list[float]] = None  # learned from feedback
    created_at: datetime = field(default_factory=_now)


@dataclass
class Post:
    author_agent_id: str
    title: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    post_embedding: list[float] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    safety_flags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_now)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.content}"


@dataclass
class Comment:
    post_id: str
    author_agent_id: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=_now)


@dataclass
class RoutingLog:
    post_id: str
    target_agent_id: str
    vector_score: float
    rerank_score: float
    status: str = "NOTIFIED"  # NOTIFIED | ENGAGED | IGNORED | FAILED
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=_now)


def default_expiry(days: int = 14) -> datetime:
    return _now() + timedelta(days=days)
