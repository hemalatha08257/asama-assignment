"""FastAPI layer (run: uvicorn asama_network.api:app --reload)."""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from .router import RouterConfig
from .service import NetworkService

app = FastAPI(title="Agent Social Network")
svc = NetworkService(cfg=RouterConfig(threshold=0.60))  # tune per embedding model


class RegisterIn(BaseModel):
    company_name: str = Field(min_length=1, max_length=255)
    industry_category: str
    webhook_url: HttpUrl
    interest_manifesto: str = Field(min_length=10)
    blocked_industries: set[str] = set()


class PostIn(BaseModel):
    author_agent_id: str
    title: str = Field(max_length=255)
    content: str


class CommentIn(BaseModel):
    author_agent_id: str
    content: str


@app.post("/api/v1/agents/register", status_code=201)
def register(body: RegisterIn):
    a = svc.register_agent(company_name=body.company_name, industry_category=body.industry_category,
                           webhook_url=str(body.webhook_url), interest_manifesto=body.interest_manifesto,
                           blocked_industries=body.blocked_industries)
    # secret is shown once; the agent uses it to verify X-Asama-Signature
    return {"agent_id": a.id, "webhook_secret": a.webhook_secret}


@app.post("/api/v1/posts", status_code=202)
def publish(body: PostIn):
    try:
        res = svc.publish_post(body.author_agent_id, body.title, body.content)
    except KeyError:
        raise HTTPException(404, "unknown author agent")
    if res["status"] == "REJECTED":
        raise HTTPException(422, {"error": "content_policy", "flags": res["flags"]})
    return res


@app.post("/api/v1/posts/{post_id}/comments", status_code=201)
def comment(post_id: str, body: CommentIn):
    try:
        c = svc.add_comment(post_id, body.author_agent_id, body.content)
    except KeyError:
        raise HTTPException(404, "unknown post or agent")
    return {"comment_id": c.id}


@app.get("/api/v1/posts/{post_id}/comments")
def list_comments(post_id: str):
    return [{"author_agent_id": c.author_agent_id, "content": c.content}
            for c in svc.store.comments.get(post_id, [])]


@app.get("/healthz")
def health():
    return {"ok": True}
