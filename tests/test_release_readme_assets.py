import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_FIGURE_SHA256 = (
    "4467f569750c110d0d86750ffc61fa7e16f13fac683bd643df55a6ca308225e7"
)


def test_readme_presents_the_governed_agent_workflow_figure():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/assets/human-governed-agent-framework.png" in readme
    assert "framework prepared by the authors" in readme


def test_readme_presents_the_validated_interface_walkthrough():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    screenshots = [
        "ui-conversation-task-center.png",
        "ui-initial-champion-registered.png",
        "ui-expert-failure-slice-review.png",
    ]
    for filename in screenshots:
        assert f"docs/assets/{filename}" in readme
        assert (ROOT / "docs" / "assets" / filename).is_file()
    assert "https://huggingface.co/datasets/LotusRosa/BFD-ML-4K" in readme


def test_release_uses_the_exact_author_framework_figure():
    figure = ROOT / "docs" / "assets" / "human-governed-agent-framework.png"
    digest = hashlib.sha256(figure.read_bytes()).hexdigest()
    assert digest == FRAMEWORK_FIGURE_SHA256
    assert not (ROOT / "docs" / "assets" / "agent-governed-workflow.svg").exists()


def test_asset_provenance_identifies_the_unmodified_framework_figure():
    provenance = (ROOT / "docs" / "assets" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "framework" in provenance.lower()
    assert "unmodified" in provenance.lower()


def test_public_agent_materials_use_product_only_evaluation_language():
    public_files = [
        ROOT / "README.md",
        ROOT / "AGENT_COMPLETE_GPU_HANDOFF.md",
        ROOT / "SUPPORTED_ENVIRONMENTS.md",
        ROOT / "ACCUMULATING_GATE_PROTOCOL.md",
        *(ROOT / "facade_agent" / "protocols").glob("*.json"),
        *(ROOT / "facade_agent" / "static").glob("*.js"),
        *(ROOT / "facade_agent" / "static").glob("*.html"),
    ]
    for path in public_files:
        public_text = path.read_text(encoding="utf-8-sig").lower()
        assert "final test" not in public_text, path
        assert "final_test" not in public_text, path

    initial_profile = json.loads(
        (ROOT / "facade_agent" / "protocols" / "initial_champion_profile_v1.json")
        .read_text(encoding="utf-8")
    )
    assert initial_profile["hardware_target"] == "local_cuda_single_gpu_worker"
    assert initial_profile["visible_gpu_count"] == 1
    assert "distributed_world_size" not in initial_profile
