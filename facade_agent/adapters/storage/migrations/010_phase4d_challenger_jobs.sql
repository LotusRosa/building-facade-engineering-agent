CREATE TABLE IF NOT EXISTS challenger_jobs (
    job_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    batch_id TEXT NOT NULL UNIQUE,
    review_version_id TEXT NOT NULL,
    parent_champion_model_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'exported', 'awaiting_result', 'result_verified', 'registered', 'failed'
    )),
    training_profile_id TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    pool_counts_json TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL UNIQUE,
    bundle_path TEXT NOT NULL,
    bundle_sha256 TEXT NOT NULL,
    bundle_size_bytes INTEGER NOT NULL CHECK (bundle_size_bytes >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id),
    FOREIGN KEY (review_version_id) REFERENCES failure_review_versions(review_version_id),
    FOREIGN KEY (parent_champion_model_id) REFERENCES model_versions(model_id)
);

CREATE INDEX IF NOT EXISTS idx_challenger_jobs_project_time
ON challenger_jobs(project_id, created_at);
