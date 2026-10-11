-- Evaluation drafts are deliberately separate from images/datasets used by training.
CREATE TABLE IF NOT EXISTS evaluation_input_images (
    image_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    batch_id TEXT NOT NULL REFERENCES maintenance_batches(batch_id),
    cohort TEXT NOT NULL CHECK(cohort IN ('current_gate','core_safety')),
    filename TEXT NOT NULL,
    filename_key TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    annotation_status TEXT NOT NULL DEFAULT 'unlabeled' CHECK(annotation_status IN ('unlabeled','complete')),
    no_defect INTEGER NOT NULL DEFAULT 0 CHECK(no_defect IN (0,1)),
    class_ids_json TEXT NOT NULL DEFAULT '[]',
    annotation_updated_at TEXT NOT NULL,
    UNIQUE(batch_id,cohort,filename_key),
    UNIQUE(batch_id,sha256)
);
