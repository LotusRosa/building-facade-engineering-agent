ALTER TABLE projects ADD COLUMN model_storage_root TEXT;
ALTER TABLE projects ADD COLUMN model_storage_locked_at TEXT;

ALTER TABLE model_versions ADD COLUMN checkpoint_relpath TEXT;
ALTER TABLE model_versions ADD COLUMN manifest_relpath TEXT;
ALTER TABLE model_versions ADD COLUMN manifest_sha256 TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_model_storage_root
ON projects(model_storage_root)
WHERE model_storage_root IS NOT NULL AND deleted_at IS NULL;
