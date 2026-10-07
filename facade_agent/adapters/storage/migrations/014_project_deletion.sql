ALTER TABLE projects ADD COLUMN deleted_at TEXT;

CREATE INDEX IF NOT EXISTS idx_projects_visible
ON projects(deleted_at, created_at);
