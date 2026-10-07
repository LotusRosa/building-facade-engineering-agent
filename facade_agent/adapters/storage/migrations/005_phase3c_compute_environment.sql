CREATE TABLE IF NOT EXISTS compute_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    configured_mode TEXT NOT NULL DEFAULT 'auto'
        CHECK (configured_mode IN ('auto', 'local_cpu', 'local_gpu', 'external_bundle')),
    gpu_devices_json TEXT NOT NULL DEFAULT '[]',
    cpu_threads INTEGER NOT NULL DEFAULT 0 CHECK (cpu_threads >= 0),
    updated_at TEXT NOT NULL
);
