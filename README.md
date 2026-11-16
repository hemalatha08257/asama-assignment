# Asama.ai Assignment: Problem 1 & Problem 2

Both solutions run and are tested **without network or API keys** (pure stdlib core, swappable adapters).
Run all tests: `cd problem1 && python3 -m unittest discover -s tests` · `cd problem2 && python3 -m unittest discover -s tests`

---
## Problem 1 -- Social network for AI agents (`problem1/`)

```
agent --POST /posts--> [safety screen] -> embed -> store -> Router
                                                     |  Stage1 filters (active, not author, blocklists, trust, expiry)
                                                     |  Stage2 ANN top-K (cosine; pgvector HNSW in prod)
                                                     |  Stage3 rerank + threshold + max_notify cap
                                                     v
                           Deliverer: HMAC-signed webhook, retries w/ backoff, circuit breaker
                                                     v
                          target agent --POST /posts/{id}/comments--> feedback loop (ENGAGED / IGNORED)
```

| File | Role |
|---|---|
| `models.py`, `store.py`, `schema.sql` | Domain objects; in-memory repo; production Postgres + pgvector schema, HNSW index, one-query Stage1+2 |
| `embeddings.py` | `ConceptHashEmbedder` (offline, bridges "Diwali food offer" <-> "UPI cashback"); `OpenAIEmbedder` adapter |
| `reranker.py` | `ConceptReranker` (pairwise: dense + concept coverage + complementarity); `CohereReranker` adapter |
| `router.py` | The 3-stage funnel, trust-weighted, configurable `top_k / threshold / max_notify` |
| `safety.py` | Unicode normalisation, injection heuristics, ML-guard hook (Llama Guard) |
| `delivery.py` | Untrusted-data envelope, HMAC-SHA256 + timestamp (replay-safe), backoff, per-target breaker |
| `feedback.py` | Engagement rate; positive/negative drift of a *dynamic* vector (manifesto never mutated) |
| `api.py`, `demo.py` | FastAPI endpoints per the spec; runnable end-to-end demo |

**Key decisions**
- **Funnel for cost/latency:** cheap deterministic filters -> vector ANN -> expensive rerank on <=50 only. LLM tokens are spent on a handful of pairs, not N agents.
- **Precision over recall by design:** a threshold plus `max_notify=5` stops spam; spam erodes trust in the network faster than a missed match.
- **Injection defence is layered:** (1) screen at ingress, (2) post delivered as `untrusted_content` with `content_trust` marker so receivers never splice it into a system prompt, (3) signed payloads, (4) trust score lowers a bad actor's reach.
- **Feedback without corrupting identity:** learned drift lives in `dynamic_embedding`, blended at `dynamic_blend` weight; reversible.
- **Delivery isolation:** slow endpoints can't block routing (breaker opens after repeated failures); `UNIQUE(post_id,target_agent_id)` makes redelivery idempotent.

**Production path:** FastAPI -> Postgres/pgvector (HNSW, `ef_search` tuned, pgvector >=0.8 iterative scans for filtered ANN) -> Redis Streams/Celery workers for embed/route/deliver -> cross-encoder or fast LLM for Stage 3. Cache post embeddings; batch rerank; shard by industry if >100k agents.

**Honest caveats:** the demo embedder/reranker are lexical-concept stand-ins, so scores aren't comparable to real models; **recalibrate the 0.75 threshold** on labelled pairs (precision@5, notification-to-comment rate) after swapping in real ones. A 100 ms budget suits a cross-encoder, not a full LLM judge over 50 candidates (use LLM only for the top ~5 or run async). `api.py` was not executed in my sandbox (FastAPI not installed); the logic it calls is fully tested.

---
## Problem 2 -- Local Linux hardware Q&A assistant (`problem2/`)

```
question -> [LLM tool-calling | deterministic router] -> allow-listed tool name
         -> Executor (no shell, abs path, fixed argv, 5s timeout, clean env, scoped sudo)
         -> Parser (block-aware, bytes, validated JSON) -> grounded answer (or "cannot determine")
```
Try: `python3 -m hwassist.cli -q "Can I upgrade my RAM?"` or interactive `python3 -m hwassist.cli`
LLM mode: `ollama pull qwen2.5:7b && python3 -m hwassist.cli --llm qwen2.5:7b` (all data stays on the laptop).

**Security model**
1. The LLM never writes commands; it picks from `TOOLS` (registry.py). Unknown tool or any extra argument -> rejected.
2. `subprocess.run(argv, shell=False)`, absolute binary path from `shutil.which`, `LANG=C`, stdin closed, output capped, 5 s timeout.
3. Root only for `dmidecode`, via `sudo -n` + `packaging/hw-assistant.sudoers`, which pins the **exact** argument list (`dmidecode -t memory`), so appended flags or chained commands are denied by sudo itself. The agent never runs as root.
4. Tool results are passed to the LLM as data; the grounding prompt forbids invented specs.

**Reliability**
- Capability discovery at startup; only installed tools are offered to the model.
- Fallback hierarchy: no `dmidecode`/no sudo -> `/proc` memory totals plus an explicit "cannot determine slots/max RAM" message, never a guess.
- Session cache: follow-ups ("and the GPU?", "the CPU again?") reuse gathered JSON and don't re-execute.
- Multi-intent questions route to the union of tools ("what cpu and what gpu?").

**Fixes to the reference blueprint (found by testing):** its `Size:\s*` regex also matches `Cache Size`/`Logical Size` lines, and the real `dmidecode` block starts with a `Handle` line, so I parse per block; its `execute_system_tool` accepted `kwargs` and silently ignored them; `sudo` was embedded in the argv without `-n` (would hang on a password prompt).

**Not verified here:** the sandbox has no `dmidecode`, `lspci` or battery, so those parsers are tested on fixtures (realistic sample output), not live hardware. One test is skipped when run as root. Ollama mode is a thin adapter I couldn't exercise without a model; the tool loop is tested with a fake LLM, including a hallucinated `bash` tool call being blocked.

---
## Interview talking points
- Why a funnel, and how you'd measure it: precision@k, notification->comment rate, p95 latency per stage, false-negative audits on dropped posts.
- Prompt injection is mitigated by *architecture* (data/instruction separation, least privilege) with detection as defence in depth, since no classifier is complete.
- Least privilege: pinned sudoers argv > root daemon > no-root fallback; trade-offs in each.
- "Cannot determine" is a feature: grounded refusal beats a plausible hallucination.
- What I'd do next: pgvector repository + integration tests, async worker split, schema-validated tool params, `jc`-style parser fallbacks across distros, eval set of 50 hardware questions.
