ALTER TABLE pending_tool_calls ADD COLUMN workflow_version TEXT;

CREATE INDEX IF NOT EXISTS idx_pending_tool_calls_workflow_version
ON pending_tool_calls(project_id, workflow_version, status);
