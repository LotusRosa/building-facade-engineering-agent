ALTER TABLE datasets
ADD COLUMN batch_id TEXT REFERENCES maintenance_batches(batch_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_datasets_one_per_maintenance_batch
ON datasets(batch_id)
WHERE batch_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_datasets_batch
ON datasets(batch_id, created_at);
