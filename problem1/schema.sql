CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE agents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  company_name VARCHAR(255) NOT NULL,
  industry_category VARCHAR(100) NOT NULL,
  webhook_url TEXT NOT NULL,
  webhook_secret_hash TEXT NOT NULL,            -- store secret in KMS/vault; hash here for lookup
  interest_manifesto TEXT NOT NULL,
  interest_embedding vector(1536),
  dynamic_embedding vector(1536),               -- feedback-learned drift
  blocked_industries TEXT[] DEFAULT '{}',
  active BOOLEAN DEFAULT TRUE,
  trust_score REAL DEFAULT 1.0,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE posts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  author_agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
  title VARCHAR(255) NOT NULL,
  content TEXT NOT NULL,
  post_embedding vector(1536),
  safety_flags TEXT[] DEFAULT '{}',
  expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE routing_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  post_id UUID REFERENCES posts(id) ON DELETE CASCADE,
  target_agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
  vector_score REAL NOT NULL,
  rerank_score REAL NOT NULL,
  status VARCHAR(20) DEFAULT 'NOTIFIED' CHECK (status IN ('NOTIFIED','ENGAGED','IGNORED','FAILED')),
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (post_id, target_agent_id)             -- idempotent delivery
);
CREATE TABLE comments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  post_id UUID REFERENCES posts(id) ON DELETE CASCADE,
  author_agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX agents_hnsw ON agents USING hnsw (interest_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
CREATE INDEX routing_logs_agent_idx ON routing_logs (target_agent_id, created_at DESC);

-- Stage 1 + Stage 2 in ONE query (hard filters + ANN top-K):
--   $1 = post_embedding, $2 = author_agent_id, $3 = author industry
-- SET LOCAL hnsw.ef_search = 100;
-- SELECT a.id, a.company_name, 1 - (a.interest_embedding <=> $1) AS vector_score
-- FROM agents a
-- WHERE a.active AND a.id <> $2 AND a.trust_score >= 0.2
--   AND NOT ($3 = ANY(a.blocked_industries))
-- ORDER BY a.interest_embedding <=> $1
-- LIMIT 50;
-- NOTE: with highly selective filters, enable pgvector>=0.8 iterative scans
-- (hnsw.iterative_scan = relaxed_order) so filtering doesn't starve results.
