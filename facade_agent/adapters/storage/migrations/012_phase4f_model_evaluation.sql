CREATE TABLE IF NOT EXISTS evaluation_jobs (
    job_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    batch_id TEXT NOT NULL UNIQUE,
    champion_model_id TEXT NOT NULL,
    challenger_model_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'exported', 'awaiting_result', 'result_verified', 'registered', 'failed'
    )),
    evaluation_profile_id TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    source_snapshot_name TEXT NOT NULL,
    source_snapshot_sha256 TEXT NOT NULL,
    split_counts_json TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL UNIQUE,
    bundle_path TEXT NOT NULL,
    bundle_sha256 TEXT NOT NULL,
    bundle_size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id),
    FOREIGN KEY (champion_model_id) REFERENCES model_versions(model_id),
    FOREIGN KEY (challenger_model_id) REFERENCES model_versions(model_id)
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
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
    FOREIGN KEY (job_id) REFERENCES evaluation_jobs(job_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id)
);

CREATE TABLE IF NOT EXISTS evaluation_run_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('status', 'progress', 'log')),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id)
);

ALTER TABLE evidence_reports ADD COLUMN source_evaluation_job_id TEXT
    REFERENCES evaluation_jobs(job_id);
ALTER TABLE evidence_reports ADD COLUMN source_evaluation_run_id TEXT
    REFERENCES evaluation_runs(run_id);
ALTER TABLE evidence_reports ADD COLUMN evaluation_profile_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_evaluation_runs_one_active_job
ON evaluation_runs(job_id)
WHERE status IN ('queued', 'running', 'cancel_requested');

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_batch_time
ON evaluation_runs(batch_id, created_at);

CREATE INDEX IF NOT EXISTS idx_evaluation_run_events_run_time
ON evaluation_run_events(run_id, event_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_report_evaluation_job
ON evidence_reports(source_evaluation_job_id)
WHERE source_evaluation_job_id IS NOT NULL;
