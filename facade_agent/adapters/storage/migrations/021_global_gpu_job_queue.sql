CREATE TABLE IF NOT EXISTS gpu_job_queue (
    queue_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    pipeline TEXT NOT NULL CHECK (pipeline IN (
        'initial_champion',
        'champion_failure_discovery',
        'challenger_update',
        'champion_challenger_evaluation'
    )),
    run_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN (
        'queued', 'starting', 'running', 'completed',
        'failed', 'cancelled', 'interrupted'
    )),
    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1)),
    error_code TEXT,
    error_message TEXT,
    lease_owner TEXT,
    lease_acquired_at TEXT,
    enqueued_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

CREATE INDEX IF NOT EXISTS idx_gpu_job_queue_status_time
ON gpu_job_queue(status, enqueued_at, queue_id);

CREATE INDEX IF NOT EXISTS idx_gpu_job_queue_project_status_time
ON gpu_job_queue(project_id, status, enqueued_at, queue_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_gpu_job_queue_one_active_lease
ON gpu_job_queue((1))
WHERE status IN ('starting', 'running');
