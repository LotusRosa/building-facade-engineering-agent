CREATE TABLE IF NOT EXISTS failure_review_versions (
    review_version_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL UNIQUE,
    reviewer_id TEXT NOT NULL,
    slice_count INTEGER NOT NULL CHECK (slice_count >= 0),
    eligible_slice_count INTEGER NOT NULL CHECK (eligible_slice_count >= 0),
    retained_member_records INTEGER NOT NULL CHECK (retained_member_records >= 0),
    retained_unique_images INTEGER NOT NULL CHECK (retained_unique_images >= 0),
    review_sha256 TEXT NOT NULL UNIQUE,
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES maintenance_batches(batch_id)
);

CREATE INDEX IF NOT EXISTS idx_failure_review_versions_batch
ON failure_review_versions(batch_id, created_at);
