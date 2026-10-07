CREATE TABLE IF NOT EXISTS evaluation_gates (
    gate_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    source_batch_id TEXT,
    gate_key TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('core_safety', 'round_gate')),
    status TEXT NOT NULL CHECK (status IN ('pending', 'active')),
    source_evaluation_job_id TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    source_split TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    image_count INTEGER NOT NULL CHECK (image_count > 0),
    created_at TEXT NOT NULL,
    activated_at TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (source_batch_id) REFERENCES maintenance_batches(batch_id),
    FOREIGN KEY (source_evaluation_job_id) REFERENCES evaluation_jobs(job_id),
    UNIQUE (project_id, gate_key),
    UNIQUE (project_id, source_batch_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_evaluation_gates_one_core
ON evaluation_gates(project_id)
WHERE role='core_safety';

CREATE INDEX IF NOT EXISTS idx_evaluation_gates_project_status
ON evaluation_gates(project_id, status, created_at);
