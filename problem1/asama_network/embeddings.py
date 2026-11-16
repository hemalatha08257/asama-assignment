"""Embedding providers.

`ConceptHashEmbedder` is an offline, deterministic stand-in so the whole system
is runnable and testable without API keys. It hashes tokens AND a small
business-concept lexicon, so "Diwali food discount" lands near "UPI cashback
on festive e-commerce" -- the semantic bridge the problem statement calls out.
In production swap in `OpenAIEmbedder` (same interface).
"""
from __future__ import annotations
import hashlib
import math
import re
from typing import Protocol

DIM = 512

CONCEPTS: dict[str, set[str]] = {
    "festive": {"diwali", "festive", "festival", "holi", "eid", "christmas", "navratri",
                "dussehra", "onam", "pongal", "sale", "season"},
    "food": {"food", "delivery", "restaurant", "meal", "meals", "dining", "grocery",
             "swiggy", "zomato", "kitchen", "order", "orders"},
    "payments": {"upi", "payment", "payments", "cashback", "wallet", "card", "cards",
                 "phonepe", "paytm", "pay", "checkout", "transaction", "transactions",
                 "merchant", "merchants", "fintech"},
    "promo": {"offer", "offers", "discount", "discounts", "promotion", "promotions",
              "deal", "deals", "flat", "off", "coupon", "reward", "rewards", "campaign"},
    "ecommerce": {"ecommerce", "e-commerce", "shopping", "retail", "marketplace",
                  "flipkart", "amazon", "cart", "shop"},
    "auto": {"bike", "bikes", "car", "cars", "vehicle", "vehicles", "scooter", "auto",
             "automobile", "ev", "motorcycle", "dealership", "bajaj"},
    "lending": {"loan", "loans", "emi", "credit", "lending", "bank", "banking",
                "mortgage", "interest", "hdfc", "financing", "finance"},
    "logistics": {"shipping", "logistics", "courier", "warehouse", "fleet", "supply"},
}
_TOKEN_TO_CONCEPTS: dict[str, list[str]] = {}
for concept, words in CONCEPTS.items():
    for w in words:
        _TOKEN_TO_CONCEPTS.setdefault(w, []).append(concept)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def concepts_in(text: str) -> set[str]:
    out: set[str] = set()
    for t in tokenize(text):
        out.update(_TOKEN_TO_CONCEPTS.get(t, ()))
    return out


def _bucket(feature: str) -> tuple[int, float]:
    h = hashlib.blake2b(feature.encode(), digest_size=8).digest()
    n = int.from_bytes(h, "big")
    return n % DIM, 1.0 if (n >> 63) & 1 else -1.0


def normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are unit vectors


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class ConceptHashEmbedder:
    def __init__(self, concept_weight: float = 2.0, token_weight: float = 0.5):
        self.cw, self.tw = concept_weight, token_weight

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for tok in tokenize(text):
            i, s = _bucket("t:" + tok)
            vec[i] += s * self.tw
            for c in _TOKEN_TO_CONCEPTS.get(tok, ()):
                j, s2 = _bucket("c:" + c)
                vec[j] += s2 * self.cw
        return normalize(vec)


class OpenAIEmbedder:  # pragma: no cover - needs network + key
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client, self.model = OpenAI(), model

    def embed(self, text: str) -> list[float]:
        r = self.client.embeddings.create(model=self.model, input=text)
        return normalize(r.data[0].embedding)
