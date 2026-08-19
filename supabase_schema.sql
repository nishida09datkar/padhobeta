-- Supabase SQL Schema for Padhobeta Session DAG System
-- Run this in your Supabase SQL Editor (https://supabase.com/dashboard/project/tzhykqxuvopqphtnbnra/sql/new)

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    document_id TEXT,
    is_active BOOLEAN DEFAULT TRUE
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- DAG Nodes table (session summaries)
CREATE TABLE IF NOT EXISTS dag_nodes (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- DAG Edges table (session relationships)
CREATE TABLE IF NOT EXISTS dag_edges (
    id BIGSERIAL PRIMARY KEY,
    parent_session_id TEXT NOT NULL REFERENCES dag_nodes(session_id) ON DELETE CASCADE,
    child_session_id TEXT NOT NULL REFERENCES dag_nodes(session_id) ON DELETE CASCADE,
    edge_type TEXT DEFAULT 'context_flow',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(parent_session_id, child_session_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_edges_parent ON dag_edges(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_edges_child ON dag_edges(child_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);

-- Enable Row Level Security (RLS) - recommended for production
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE dag_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE dag_edges ENABLE ROW LEVEL SECURITY;

-- Policies for anon access (matching your anon key permissions)
CREATE POLICY "Allow all operations for anon" ON sessions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all operations for anon" ON messages FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all operations for anon" ON dag_nodes FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all operations for anon" ON dag_edges FOR ALL USING (true) WITH CHECK (true);
