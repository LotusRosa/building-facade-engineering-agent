CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('initial_training', 'screening', 'maintenance')),
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'validated', 'frozen')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, name),
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

CREATE TABLE IF NOT EXISTS images (
    image_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    mime_type TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    health_status TEXT NOT NULL CHECK (health_status IN ('ok', 'corrupt', 'unsupported')),
    imported_at TEXT NOT NULL,
    UNIQUE(dataset_id, filename),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

CREATE TABLE IF NOT EXISTS image_annotations (
    image_id TEXT PRIMARY KEY,
    no_defect INTEGER NOT NULL DEFAULT 0 CHECK (no_defect IN (0, 1)),
    status TEXT NOT NULL DEFAULT 'unlabeled' CHECK (status IN ('unlabeled', 'complete')),
    updated_at TEXT NOT NULL,
    FOREIGN KEY (image_id) REFERENCES images(image_id)
);

CREATE TABLE IF NOT EXISTS image_annotation_labels (
    image_id TEXT NOT NULL,
    class_id TEXT NOT NULL,
    PRIMARY KEY (image_id, class_id),
    FOREIGN KEY (image_id) REFERENCES images(image_id),
    FOREIGN KEY (class_id) REFERENCES classes(class_id)
);

CREATE TABLE IF NOT EXISTS label_versions (
    label_version_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    labels_sha256 TEXT NOT NULL,
    image_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(dataset_id, version_number),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

CREATE INDEX IF NOT EXISTS idx_datasets_project_role
ON datasets(project_id, role, created_at);

CREATE INDEX IF NOT EXISTS idx_images_dataset
ON images(dataset_id, imported_at);

CREATE INDEX IF NOT EXISTS idx_images_dataset_sha256
ON images(dataset_id, sha256);

CREATE INDEX IF NOT EXISTS idx_annotation_labels_class
ON image_annotation_labels(class_id, image_id);

