CREATE TABLE IF NOT EXISTS training_runs (
    run_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
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
    error_message TEXT,
    pid INTEGER,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES training_jobs(job_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

CREATE TABLE IF NOT EXISTS training_run_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('status', 'progress', 'log')),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
);

ALTER TABLE model_versions ADD COLUMN source_training_job_id TEXT
    REFERENCES training_jobs(job_id);
ALTER TABLE model_versions ADD COLUMN source_training_run_id TEXT
    REFERENCES training_runs(run_id);
ALTER TABLE model_versions ADD COLUMN result_bundle_sha256 TEXT;
ALTER TABLE model_versions ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}';

CREATE UNIQUE INDEX IF NOT EXISTS idx_training_runs_one_active_job
ON training_runs(job_id)
WHERE status IN ('queued', 'running', 'cancel_requested');

CREATE INDEX IF NOT EXISTS idx_training_runs_project_time
ON training_runs(project_id, created_at);

CREATE INDEX IF NOT EXISTS idx_training_run_events_run_time
ON training_run_events(run_id, event_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_initial_champion_job
ON model_versions(source_training_job_id)
WHERE source_training_job_id IS NOT NULL AND role = 'champion';
