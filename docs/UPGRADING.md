# Update the program while preserving projects

[中文](UPGRADING_ZH.md)

Extract the new software into a separate directory. Retain the original software, runtime, photos, and checkpoints. The public product name is `building-facade-engineering-agent`; existing commands continue to use `facade_next`.

## Existing facade_next projects

Finish active requests and do not run two services against the same writable runtime. Open the original state with the new program:

```bash
python -m facade_next serve --state /absolute/path/to/original/runtime --port 8766 --ui-only
```

Keep the original host's data/checkpoint paths unchanged. Use **Saved tasks** to inspect frozen configuration, existing evaluation, and recorded decisions without retraining or adopting again. For a consistent backup, stop the service after Workers finish and copy the whole runtime, rather than just its database file.

## Runtime compatibility

Open only a compatible `facade_next` state directory. Other database or checkpoint formats require explicit provenance and supported conversion; renaming a file or folder does not migrate it. Frozen absolute paths are not silently remapped between Windows and Ubuntu. Keep the data and checkpoint paths referenced by existing jobs.

## Active workflow

Use the project's selected adaptation strategies, independent Val and acceptance data, and one Worker per selected GPU. Current labels and annotation semantics are documented in [data preparation](DATA_PREPARATION.md). Recorded screenshots retain their actual session pixels and are not relabeled as a new run.
