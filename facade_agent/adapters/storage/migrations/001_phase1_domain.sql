CREATE TABLE IF NOT EXISTS agent_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    active_project_id TEXT,
    locale TEXT NOT NULL DEFAULT 'zh-CN',
    updated_at TEXT NOT NULL,
    FOREIGN KEY (active_project_id) REFERENCES projects(project_id)
);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    state TEXT NOT NULL,
    active_batch_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classes (
    class_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    display_name_zh TEXT,
    display_name_en TEXT,
    description TEXT NOT NULL DEFAULT '',
    sort_order INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at TEXT NOT NULL,
    UNIQUE(project_id, display_name COLLATE NOCASE),
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

CREATE TABLE IF NOT EXISTS maintenance_batches (
    batch_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    state TEXT NOT NULL,
    formal_decision TEXT CHECK (formal_decision IN ('deploy', 'hold') OR formal_decision IS NULL),
    decision_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, name),
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    parent_model_id TEXT,
    source_batch_id TEXT,
    role TEXT NOT NULL CHECK (role IN ('champion', 'challenger', 'archived')),
    checkpoint_path TEXT NOT NULL,
    checkpoint_sha256 TEXT NOT NULL,
    thresholds_json TEXT NOT NULL,
    training_profile_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (parent_model_id) REFERENCES model_versions(model_id),
    FOREIGN KEY (source_batch_id) REFERENCES maintenance_batches(batch_id)
);

CREATE TABLE IF NOT EXISTS active_models (
    project_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (model_id) REFERENCES model_versions(model_id)
);

CREATE TABLE IF NOT EXISTS failure_slices (
    slice_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    class_id TEXT NOT NULL,
    error_type TEXT NOT NULL CHECK (error_type IN ('FP', 'FN')),
    support INTEGER NOT NULL CHECK (support > 0),
    clustering_profile_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id),
    FOREIGN KEY (class_id) REFERENCES classes(class_id)
);

CREATE TABLE IF NOT EXISTS failure_slice_members (
    slice_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    membership_rank INTEGER NOT NULL,
    PRIMARY KEY (slice_id, image_id),
    FOREIGN KEY (slice_id) REFERENCES failure_slices(slice_id)
);

CREATE TABLE IF NOT EXISTS slice_reviews (
    slice_id TEXT PRIMARY KEY,
    decision TEXT NOT NULL CHECK (decision IN ('accept', 'trim', 'reject')),
    excluded_json TEXT NOT NULL DEFAULT '[]',
    reviewer_id TEXT NOT NULL,
    knowledge_version TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (slice_id) REFERENCES failure_slices(slice_id)
);

CREATE TABLE IF NOT EXISTS tool_runs (
    run_id TEXT PRIMARY KEY,
    project_id TEXT,
    batch_id TEXT,
    tool_name TEXT NOT NULL,
    status TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id)
);

CREATE TABLE IF NOT EXISTS evidence_reports (
    evidence_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    batch_id TEXT NOT NULL UNIQUE,
    champion_model_id TEXT NOT NULL,
    challenger_model_id TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    artifact_path TEXT,
    artifact_sha256 TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id)
);

CREATE TABLE IF NOT EXISTS deployment_decisions (
    decision_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL UNIQUE,
    decision TEXT NOT NULL CHECK (decision IN ('deploy', 'hold')),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    actor_id TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id)
);

CREATE TABLE IF NOT EXISTS artifact_refs (
    artifact_id TEXT PRIMARY KEY,
    project_id TEXT,
    batch_id TEXT,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE(path, sha256),
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id)
);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at TEXT NOT NULL,
    project_id TEXT,
    batch_id TEXT,
    actor_type TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT,
    payload_json TEXT NOT NULL,
    previous_event_sha256 TEXT,
    event_sha256 TEXT NOT NULL UNIQUE,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id)
);

CREATE INDEX IF NOT EXISTS idx_classes_project ON classes(project_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_batches_project ON maintenance_batches(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_models_project ON model_versions(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_slices_batch ON failure_slices(batch_id, class_id, error_type);
CREATE INDEX IF NOT EXISTS idx_tool_runs_scope ON tool_runs(project_id, batch_id, started_at);
CREATE INDEX IF NOT EXISTS idx_audit_scope ON audit_events(project_id, batch_id, event_id);

