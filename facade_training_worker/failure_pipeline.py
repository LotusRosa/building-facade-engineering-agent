from __future__ import annotations

import json
import os
import shutil
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from . import WORKER_PROTOCOL_VERSION
from .bundle import VerifiedBundle
from .data import extract_images, parse_classes, parse_image_records
from .inference import distributed_inference
from .result_io import csv_bytes, json_bytes, write_result_zip


def consensus_slices(
    failures: list[dict[str, Any]],
    features: dict[str, list[float]],
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = profile["clustering"]
    slices: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in failures:
        grouped.setdefault((row["class_id"], row["error_type"]), []).append(row)
    slice_number = 0
    for (class_id, error_type), rows in sorted(grouped.items()):
        if len(rows) < int(config["min_failures_per_class_error_pool"]):
            continue
        rows = sorted(rows, key=lambda item: item["image_id"])
        matrix = normalize(np.asarray([features[row["image_id"]] for row in rows], dtype=np.float64))
        components = min(int(profile["projection"]["pca_components_max"]), len(rows) - 1, matrix.shape[1])
        if components >= 2 and components < matrix.shape[1]:
            matrix = PCA(n_components=components, random_state=int(config["cluster_seed"])).fit_transform(matrix)
        matrix = normalize(matrix)
        maximum_k = min(int(config["max_clusters"]), len(rows) // int(config["min_cluster_size"]))
        candidates: list[tuple[float, float, int, np.ndarray, np.ndarray]] = []
        for k in range(2, maximum_k + 1):
            labels = [
                KMeans(
                    n_clusters=k,
                    n_init=int(config["kmeans_n_init_per_run"]),
                    random_state=int(config["cluster_seed"]) + run,
                ).fit_predict(matrix)
                for run in range(int(config["runs"]))
            ]
            mean_ari = float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]))
            if mean_ari < float(config["min_mean_pairwise_ari"]):
                continue
            coassociation = np.mean([label[:, None] == label[None, :] for label in labels], axis=0)
            consensus = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(1.0 - coassociation)
            silhouette = float(silhouette_score(matrix, consensus, metric="euclidean"))
            if silhouette >= float(config["min_silhouette"]):
                candidates.append((silhouette, mean_ari, -k, consensus, coassociation))
        if not candidates:
            continue
        silhouette, mean_ari, negative_k, consensus, coassociation = max(candidates, key=lambda item: item[:3])
        for cluster_id in sorted(set(int(value) for value in consensus)):
            indices = np.flatnonzero(consensus == cluster_id)
            stable: list[tuple[float, int]] = []
            for index in indices:
                peers = indices[indices != index]
                score = float(coassociation[index, peers].mean()) if len(peers) else 1.0
                if score >= float(config["stable_member_coassociation"]):
                    stable.append((score, int(index)))
            if len(stable) < int(config["min_cluster_size"]):
                continue
            stable.sort(key=lambda item: (-item[0], -float(rows[item[1]]["severity"]), rows[item[1]]["image_id"]))
            slice_number += 1
            key = f"{class_id}-{error_type.lower()}-{slice_number:03d}"
            consensus_score = float(np.mean([item[0] for item in stable]))
            slices.append({
                "slice_key": key,
                "class_id": class_id,
                "error_type": error_type,
                "support": len(stable),
                "consensus_score": consensus_score,
                "representative_image_id": rows[stable[0][1]]["image_id"],
                "silhouette": silhouette,
                "mean_pairwise_ari": mean_ari,
                "cluster_count": -negative_k,
            })
            for rank, (_, index) in enumerate(stable, 1):
                members.append({"slice_key": key, "image_id": rows[index]["image_id"], "membership_rank": rank})
    return slices, members


def run_failure_discovery(
    *, bundle: VerifiedBundle, output: Path, run_id: str, context: Any, emit: Callable[..., None]
) -> Path | None:
    profile = bundle.read_json("discovery_profile.json")
    champion = bundle.read_json("champion.json")
    class_ids = parse_classes(bundle.read_json("classes.json"))
    records = parse_image_records(bundle.read_jsonl("labels.jsonl"), class_ids, bundle.names)
    workspace = output / ".failure_discovery_workspace"
    if context.rank == 0:
        workspace.mkdir()
        image_paths = extract_images(bundle.path, records, workspace / "images")
        checkpoint = workspace / "champion.pt"
        with zipfile.ZipFile(bundle.path) as archive:
            checkpoint.write_bytes(archive.read("champion/checkpoint.pt"))
        (workspace / "paths.json").write_text(json.dumps({key: str(value) for key, value in image_paths.items()}, sort_keys=True), encoding="utf-8")
    context.dist.barrier()
    image_paths = {key: Path(value) for key, value in json.loads((workspace / "paths.json").read_text(encoding="utf-8")).items()}
    inferred = distributed_inference(
        records=records, image_paths=image_paths, class_ids=class_ids,
        checkpoint=workspace / "champion.pt", input_size=int(profile["input_size"]), context=context,
        include_features=True,
    )
    result: Path | None = None
    if context.rank == 0:
        assert inferred is not None
        rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        feature_map = {image_id: item["features"] for image_id, item in inferred.items()}
        for record in sorted(records, key=lambda item: item.image_id):
            item = inferred[record.image_id]
            truth = set(record.class_ids)
            for index, class_id in enumerate(class_ids):
                label = int(class_id in truth)
                score = float(item["probabilities"][index])
                threshold = float(champion["thresholds"][class_id])
                predicted = int(score >= threshold)
                error_type = "TP" if label and predicted else "FN" if label else "FP" if predicted else "TN"
                severity = abs(score - threshold) if error_type in {"FP", "FN"} else 0.0
                row = {"image_id": record.image_id, "class_id": class_id, "label": label, "score": score,
                       "threshold": threshold, "predicted": predicted, "error_type": error_type, "severity": severity}
                rows.append(row)
                if error_type in {"FP", "FN"}:
                    failures.append(row)
        slices, members = consensus_slices(failures, feature_map, profile)
        manifest = {
            "schema_version": 1, "worker_protocol_version": WORKER_PROTOCOL_VERSION,
            "job_id": bundle.manifest["job_id"], "run_id": run_id,
            "content_fingerprint": bundle.manifest["content_fingerprint"],
            "champion_model_id": champion["model_id"],
            "champion_checkpoint_sha256": champion["checkpoint_sha256"],
            "discovery_profile_id": profile["profile_id"], "thresholds": champion["thresholds"],
            "thresholds_retuned": False, "discovery_passes": 1,
            "prediction_rows": len(rows), "failure_records": len(failures), "failure_slices": len(slices),
        }
        result = write_result_zip(output, {
            "result_manifest.json": json_bytes(manifest),
            "predictions.csv": csv_bytes(["image_id", "class_id", "label", "score", "threshold", "predicted", "error_type", "severity"], rows),
            "failure_slices.json": json_bytes(slices),
            "failure_slice_members.csv": csv_bytes(["slice_key", "image_id", "membership_rank"], members),
        })
        emit("progress", stage="failure_discovery_complete", images_completed=len(records), images_total=len(records), percent=100)
    context.dist.barrier()
    if context.rank == 0:
        shutil.rmtree(workspace)
    context.dist.barrier()
    return result
