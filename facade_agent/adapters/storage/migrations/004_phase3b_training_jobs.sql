CREATE TABLE IF NOT EXISTS training_jobs (
    job_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    label_version_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('initial_champion')),
    status TEXT NOT NULL CHECK (status IN ('exported', 'awaiting_result', 'result_verified', 'registered', 'failed')),
    training_profile_id TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL UNIQUE,
    bundle_path TEXT NOT NULL,
    bundle_sha256 TEXT NOT NULL,
    bundle_size_bytes INTEGER NOT NULL CHECK (bundle_size_bytes >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    FOREIGN KEY (label_version_id) REFERENCES label_versions(label_version_id)
);

CREATE INDEX IF NOT EXISTS idx_training_jobs_project_time
ON training_jobs(project_id, created_at);
