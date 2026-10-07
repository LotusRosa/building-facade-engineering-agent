CREATE TABLE IF NOT EXISTS llm_messages (
    message_id TEXT PRIMARY KEY,
    project_id TEXT,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

CREATE TABLE IF NOT EXISTS pending_tool_calls (
    pending_id TEXT PRIMARY KEY,
    project_id TEXT,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'failed')),
    requested_at TEXT NOT NULL,
    resolved_at TEXT,
    resolved_by TEXT,
    result_json TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

CREATE INDEX IF NOT EXISTS idx_llm_messages_project_time
ON llm_messages(project_id, created_at);

CREATE INDEX IF NOT EXISTS idx_pending_tool_calls_status
ON pending_tool_calls(status, requested_at);
