CREATE TABLE IF NOT EXISTS final_test_image_inventory (
    project_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    source_artifact_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (project_id, image_id),
    UNIQUE (project_id, content_sha256),
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

CREATE INDEX IF NOT EXISTS idx_final_test_inventory_project_hash
ON final_test_image_inventory(project_id, content_sha256);
