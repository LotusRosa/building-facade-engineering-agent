# Update the program while preserving projects

[中文](UPGRADING_ZH.md)

Extract the new software into a separate directory. Retain the original software, runtime, photos, and checkpoints. The public product name is `building-facade-engineering-agent`; existing commands continue to use `facade_next`.

## Existing facade_next projects

Finish active requests and do not run two services against the same writable runtime. Open the original state with the new program:

```bash
python -m facade_next serve --state /absolute/path/to/original/runtime --port 8766 --ui-only
```

Keep the original host's data/checkpoint paths unchanged. Use **Saved tasks** to inspect frozen configuration, existing evaluation, and recorded decisions without retraining or adopting again. For a consistent backup, stop the service after Workers finish and copy the whole runtime, rather than just its database file.

## Earlier facade_agent implementation

The earlier public code used `facade_agent`/`facade_training_worker` and a different database/model format. Its database is not an interchangeable `facade_next` runtime. This release does not automatically migrate that schema.

The complete previous implementation remains at [its original commit](https://github.com/LotusRosa/building-facade-engineering-agent/tree/dcd90cb551ec3e59b61e8188d998a898437a709b). Continue an old project with its original implementation, or create a separate current project and import the verified pre-split data. Checkpoint conversion requires supported contracts, preprocessing, and original role provenance; changing a filename is insufficient. Absolute-path tasks are not silently remapped across Windows and Ubuntu.

Current adaptation no longer requires failure-slice clustering and expert slice review before training. It uses the engineer's chosen strategies with independent Val and acceptance data. The previous single-global-Worker restriction is replaced by the selected per-GPU slots. Current classes and annotation semantics are documented in [data preparation](DATA_PREPARATION.md); old dynamic-taxonomy and keyboard-shortcut behavior should not be assumed.

Earlier screenshots remain in Git history. The README now shows actual, newer Ubuntu validation screenshots, with their recording context stated rather than relabeling historical images as newly captured output.
