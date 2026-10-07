CREATE TABLE IF NOT EXISTS screening_jobs (
    job_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    batch_id TEXT NOT NULL UNIQUE,
    dataset_id TEXT NOT NULL,
    label_version_id TEXT NOT NULL,
    champion_model_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'exported', 'awaiting_result', 'result_verified', 'failed'
    )),
    discovery_profile_id TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL UNIQUE,
    bundle_path TEXT NOT NULL,
    bundle_sha256 TEXT NOT NULL,
    bundle_size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    FOREIGN KEY (label_version_id) REFERENCES label_versions(label_version_id),
    FOREIGN KEY (champion_model_id) REFERENCES model_versions(model_id)
);

CREATE TABLE IF NOT EXISTS screening_runs (
    run_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'queued', 'running', 'cancel_requested', 'cancelled',
        'result_verified', 'failed', 'interrupted'
    )),
    worker_protocol_version INTEGER NOT NULL,
    gpu_devices_json TEXT NOT NULL,
    world_size INTEGER NOT NULL CHECK (world_size > 0),
    preflight_json TEXT NOT NULL,
    command_json TEXT NOT NULL,
    progress_json TEXT NOT NULL DEFAULT '{}',
    log_path TEXT NOT NULL,
    result_bundle_path TEXT,
    result_bundle_sha256 TEXT,
    summary_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    pid INTEGER,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES screening_jobs(job_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id)
);

CREATE TABLE IF NOT EXISTS screening_run_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('status', 'progress', 'log')),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES screening_runs(run_id)
);

ALTER TABLE failure_slices ADD COLUMN source_screening_run_id TEXT
    REFERENCES screening_runs(run_id);
ALTER TABLE failure_slices ADD COLUMN source_slice_key TEXT;
ALTER TABLE failure_slices ADD COLUMN consensus_score REAL;
ALTER TABLE failure_slices ADD COLUMN representative_image_id TEXT
    REFERENCES images(image_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_screening_runs_one_active_job
ON screening_runs(job_id)
WHERE status IN ('queued', 'running', 'cancel_requested');

CREATE INDEX IF NOT EXISTS idx_screening_runs_batch_time
ON screening_runs(batch_id, created_at);

CREATE INDEX IF NOT EXISTS idx_screening_run_events_run_time
ON screening_run_events(run_id, event_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_failure_slices_source_key
ON failure_slices(source_screening_run_id, source_slice_key)
WHERE source_screening_run_id IS NOT NULL;
