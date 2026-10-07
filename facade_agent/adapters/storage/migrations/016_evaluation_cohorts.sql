CREATE TABLE IF NOT EXISTS evaluation_cohorts (
    cohort_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    source_batch_id TEXT NOT NULL,
    source_evaluation_job_id TEXT NOT NULL,
    source_round_name TEXT NOT NULL,
    origin_role TEXT NOT NULL CHECK (origin_role IN (
        'core_safety_seed', 'core_safety_addition', 'current_gate'
    )),
    status TEXT NOT NULL CHECK (status IN ('pending', 'active')),
    artifact_path TEXT NOT NULL,
    source_split TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    image_count INTEGER NOT NULL CHECK (image_count > 0),
    label_version_id TEXT,
    taxonomy_sha256 TEXT,
    created_at TEXT NOT NULL,
    activated_at TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (source_batch_id) REFERENCES maintenance_batches(batch_id),
    FOREIGN KEY (source_evaluation_job_id) REFERENCES evaluation_jobs(job_id),
    UNIQUE (project_id, source_batch_id, origin_role),
    UNIQUE (project_id, source_evaluation_job_id, origin_role)
);

CREATE INDEX IF NOT EXISTS idx_evaluation_cohorts_project_status
ON evaluation_cohorts(project_id, status, created_at, cohort_id);

CREATE INDEX IF NOT EXISTS idx_evaluation_cohorts_project_origin
ON evaluation_cohorts(project_id, origin_role, created_at, cohort_id);

CREATE TABLE IF NOT EXISTS round_completion_summaries (
    batch_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('retain', 'promote')),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    previous_champion_model_id TEXT NOT NULL,
    active_champion_model_id TEXT NOT NULL,
    challenger_model_id TEXT NOT NULL,
    current_gate_cohort_id TEXT NOT NULL,
    core_safety_cohort_id TEXT NOT NULL,
    cumulative_core_safety_images INTEGER NOT NULL CHECK (cumulative_core_safety_images > 0),
    evidence_id TEXT NOT NULL,
    final_test_read INTEGER NOT NULL DEFAULT 0 CHECK (final_test_read IN (0, 1)),
    completed_at TEXT NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (previous_champion_model_id) REFERENCES model_versions(model_id),
    FOREIGN KEY (active_champion_model_id) REFERENCES model_versions(model_id),
    FOREIGN KEY (challenger_model_id) REFERENCES model_versions(model_id),
    FOREIGN KEY (current_gate_cohort_id) REFERENCES evaluation_cohorts(cohort_id),
    FOREIGN KEY (core_safety_cohort_id) REFERENCES evaluation_cohorts(cohort_id),
    FOREIGN KEY (evidence_id) REFERENCES evidence_reports(evidence_id)
);

INSERT OR IGNORE INTO evaluation_cohorts(
    cohort_id, project_id, source_batch_id, source_evaluation_job_id,
    source_round_name, origin_role, status, artifact_path, source_split,
    content_sha256, image_count, label_version_id, taxonomy_sha256,
    created_at, activated_at
)
SELECT
    'cohort_legacy_' || g.gate_id,
    g.project_id,
    COALESCE(g.source_batch_id, j.batch_id),
    g.source_evaluation_job_id,
    COALESCE(b.name, 'Legacy evaluation'),
    CASE g.role
        WHEN 'core_safety' THEN 'core_safety_seed'
        ELSE 'current_gate'
    END,
    g.status,
    g.artifact_path,
    g.source_split,
    g.content_sha256,
    g.image_count,
    NULL,
    NULL,
    g.created_at,
    g.activated_at
FROM evaluation_gates AS g
JOIN evaluation_jobs AS j ON j.job_id = g.source_evaluation_job_id
LEFT JOIN maintenance_batches AS b
    ON b.batch_id = COALESCE(g.source_batch_id, j.batch_id);
