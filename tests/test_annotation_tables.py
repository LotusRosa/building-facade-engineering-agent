from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import pytest

from facade_agent.application.annotation_tables import AnnotationTableService, parse_labels, read_table, write_table
from facade_agent.application.agent_runtime import AgentRuntime
from facade_agent.application.workflow import WorkflowSnapshotService, register_workflow_tools
from facade_agent.adapters.llm import LLMManager
from facade_agent.storage import Store
from facade_agent.tools import build_phase1_registry


@pytest.fixture
def setup_dataset(request):
    temporary = tempfile.TemporaryDirectory()
    request.addfinalizer(temporary.cleanup)
    tmp_path = Path(temporary.name)
    store = Store(tmp_path / "agent.sqlite3")
    project = store.create_project(project_name="table test", class_names=["crack", "spalling"])["project"]
    store.transition_project(project_id=project["project_id"], action="confirm_taxonomy", confirmed=True)
    dataset = store.create_dataset(project_id=project["project_id"], name="initial", role="initial_training")["dataset"]
    for index, name in enumerate(["one.png", "two.png", "三.png"]):
        store.register_image(dataset_id=dataset["dataset_id"], filename=name, stored_path=str(tmp_path / name), sha256=str(index) * 64, size_bytes=24, mime_type="image/png", width=16, height=12, health_status="ok")
    return store, project, dataset, AnnotationTableService(store)


def valid_rows():
    return [["filename", "crack", "spalling", "no_defect"], ["one.png", 1, 1, 0], ["two.png", 0, 0, 1], ["三.png", 1, 0, 0]]


@pytest.mark.parametrize("format", ["xlsx", "csv"])
def test_roundtrip_preview_confirm_and_skip_manual(setup_dataset, format):
    store, project, dataset, service = setup_dataset
    content = write_table(valid_rows(), format)
    preview = service.import_table(dataset["dataset_id"], content, f"labels.{format}")
    assert preview["imported_labels"] == 3
    assert not preview["applied"]
    assert all(i["annotation_status"] != "complete" for i in store.list_images(dataset["dataset_id"]))
    result = service.import_table(dataset["dataset_id"], content, f"labels.{format}", confirmed=True, preview_token=preview["preview_token"])
    assert result["validation"]["valid"]
    assert store.get_project(project["project_id"])["state"] == "DATA_VALIDATED"
    assert read_table(service.export(project["project_id"], dataset["dataset_id"], format), f"labels.{format}") == [[str(v) for v in row] for row in valid_rows()]
    with pytest.raises(PermissionError, match="read-only"):
        service.import_table(dataset["dataset_id"], content, f"labels.{format}")
    assert store.verify_audit_chain()["valid"]


def test_partial_blank_is_not_negative(setup_dataset):
    store, project, dataset, service = setup_dataset
    rows = valid_rows(); rows[-1][1:] = ["", "", ""]
    content = write_table(rows, "xlsx")
    preview = service.import_table(dataset["dataset_id"], content, "labels.xlsx")
    assert preview["remaining_unlabeled"] == 1
    service.import_table(dataset["dataset_id"], content, "labels.xlsx", confirmed=True, preview_token=preview["preview_token"])
    assert store.get_project(project["project_id"])["state"] == "DATA_IMPORTED"
    assert store.list_images(dataset["dataset_id"])[-1]["annotation_status"] != "complete"


@pytest.mark.parametrize("bad_values", [[0, 0, 0], [1, 0, 1], [1, "", 0], ["yes", 0, 0], [2, 0, 0]])
def test_invalid_labels_never_partially_apply(setup_dataset, bad_values):
    store, _, dataset, service = setup_dataset
    rows = valid_rows(); rows[-1][1:] = bad_values
    with pytest.raises(ValueError):
        service.import_table(dataset["dataset_id"], write_table(rows, "csv"), "labels.csv")
    assert all(i["annotation_status"] != "complete" for i in store.list_images(dataset["dataset_id"]))


@pytest.mark.parametrize("kind", ["missing", "duplicate", "headers"])
def test_filename_and_taxonomy_errors(setup_dataset, kind):
    _, _, dataset, service = setup_dataset
    rows = valid_rows()
    if kind == "missing": rows[-1][0] = "absent.png"
    elif kind == "duplicate": rows[-1][0] = "one.png"
    else: rows[0][1] = "hollow"
    with pytest.raises(ValueError):
        service.import_table(dataset["dataset_id"], write_table(rows, "csv"), "labels.csv")


def test_stale_preview_and_missing_confirmation_token(setup_dataset):
    store, _, dataset, service = setup_dataset
    content = write_table(valid_rows(), "csv")
    preview = service.import_table(dataset["dataset_id"], content, "labels.csv")
    with pytest.raises(PermissionError):
        service.import_table(dataset["dataset_id"], content, "labels.csv", confirmed=True)
    image = store.list_images(dataset["dataset_id"])[0]
    store.save_annotation(image_id=image["image_id"], class_ids=[], no_defect=True)
    with pytest.raises(PermissionError, match="changed"):
        service.import_table(dataset["dataset_id"], content, "labels.csv", confirmed=True, preview_token=preview["preview_token"])
    assert store.get_image(image["image_id"])["no_defect"]


def test_cross_project_export_rejected(setup_dataset):
    store, _, dataset, service = setup_dataset
    other = store.create_project(project_name="other", class_names=["a"])["project"]
    with pytest.raises(PermissionError):
        service.export(other["project_id"], dataset["dataset_id"], "xlsx")


def test_custom_unicode_and_reserved_column_names():
    classes = [{"class_id": "1", "display_name": "no_defect"}, {"class_id": "2", "display_name": "裂缝"}]
    rows = [["filename", "class:no_defect", "class:裂缝", "no_defect"], ["中文.png", 1, 1, 0]]
    parsed = parse_labels(write_table(rows, "xlsx"), "labels.xlsx", classes, [{"image_id": "image", "filename": "中文.png"}])
    assert parsed["entries"][0]["class_ids"] == ["1", "2"]


def test_formulas_rejected_and_xlsx_export_literals():
    rows = [["filename", "a", "no_defect"], ["=bad.png", 1, 0]]
    assert read_table(write_table(rows, "xlsx"), "labels.xlsx")[1][0] == "=bad.png"
    with pytest.raises(ValueError, match="Excel export"):
        write_table(rows, "csv")
    original = write_table(rows, "xlsx")
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(buffer, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name.endswith("sheet1.xml"):
                data = data.replace(b'<is>', b'<f>1+1</f><is>', 1)
            target.writestr(name, data)
    with pytest.raises(ValueError, match="formulas"):
        read_table(buffer.getvalue(), "labels.xlsx")


def test_chat_counts_and_correct_next_action(setup_dataset):
    store, project, dataset, _ = setup_dataset
    workflow = WorkflowSnapshotService(store)
    registry = build_phase1_registry(store); register_workflow_tools(registry, workflow)
    runtime = AgentRuntime(store, registry, LLMManager(), workflow)
    result = runtime.chat(text="标注完成了吗？", project_id=project["project_id"])
    assert result["annotation_progress"]["unlabeled_count"] == 3
    assert "Excel/CSV" in result["message"]
    assert workflow.get(project["project_id"])["next_actions"][0]["action_id"] == "review_initial_labels"
    for image in store.list_images(dataset["dataset_id"]):
        store.save_annotation(image_id=image["image_id"], class_ids=[], no_defect=True)
    result = runtime.chat(text="annotation status", project_id=project["project_id"])
    assert result["annotation_progress"]["complete"]
    assert "validation" in result["message"]
    next_action = workflow.get(project["project_id"])["next_actions"][0]
    assert next_action["tool_name"] == "validate_dataset"
    assert next_action["requires_confirmation"]


def test_stdlib_xlsx_is_readable_by_excel_library():
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.load_workbook(io.BytesIO(write_table(valid_rows(), "xlsx")))
    assert list(workbook.active.values) == [tuple(str(v) for v in row) for row in valid_rows()]


def test_split_guidance_is_optional_and_distinguishes_initial_from_recurring():
    initial = AgentRuntime._data_split_guidance(initial=True, english=True)
    recurring = AgentRuntime._data_split_guidance(initial=False, english=True)
    assert 'Train : initial Core Safety' in initial
    assert 'Train : Current Gate' in recurring
    for message in (initial, recurring):
        assert '7:3 or 8:2' in message and 'not mandatory' in message
        assert 'evaluation images never enter Train' in message
