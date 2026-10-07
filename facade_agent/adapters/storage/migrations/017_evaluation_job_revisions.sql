PRAGMA foreign_keys=OFF;
PRAGMA legacy_alter_table=ON;

ALTER TABLE evaluation_jobs RENAME TO evaluation_jobs_legacy;

CREATE TABLE evaluation_jobs (
    job_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
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
    revision_number INTEGER NOT NULL DEFAULT 1 CHECK (revision_number > 0),
    is_current INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0, 1)),
    supersedes_job_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id),
    FOREIGN KEY (champion_model_id) REFERENCES model_versions(model_id),
    FOREIGN KEY (challenger_model_id) REFERENCES model_versions(model_id),
    FOREIGN KEY (supersedes_job_id) REFERENCES evaluation_jobs(job_id),
    UNIQUE (batch_id, revision_number),
    UNIQUE (supersedes_job_id)
);

INSERT INTO evaluation_jobs(
    job_id, project_id, batch_id, champion_model_id, challenger_model_id,
    status, evaluation_profile_id, profile_json, source_snapshot_name,
    source_snapshot_sha256, split_counts_json, content_fingerprint,
    bundle_path, bundle_sha256, bundle_size_bytes, revision_number,
    is_current, supersedes_job_id, created_at, updated_at
)
SELECT
    job_id, project_id, batch_id, champion_model_id, challenger_model_id,
    status, evaluation_profile_id, profile_json, source_snapshot_name,
    source_snapshot_sha256, split_counts_json, content_fingerprint,
    bundle_path, bundle_sha256, bundle_size_bytes, 1, 1, NULL,
    created_at, updated_at
FROM evaluation_jobs_legacy;

DROP TABLE evaluation_jobs_legacy;

CREATE UNIQUE INDEX idx_evaluation_jobs_one_current_batch
ON evaluation_jobs(batch_id)
WHERE is_current=1;

CREATE INDEX idx_evaluation_jobs_project_time
ON evaluation_jobs(project_id, created_at);

PRAGMA legacy_alter_table=OFF;
PRAGMA foreign_keys=ON;
