from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import pytest

from facade_agent.application.annotation_tables import read_table, write_table
from facade_agent.application.evaluation_input import EvaluationInputService
from facade_agent.application.workflow import WorkflowSnapshotService
import test_phase4f_model_evaluation as evaluation_tests


def png(seed: int) -> bytes:
    def chunk(name, payload):
        return struct.pack('>I', len(payload)) + name + payload + struct.pack('>I', zlib.crc32(name + payload))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(bytes([0, seed % 256, 0, 0]))) + chunk(b'IEND', b''))


@pytest.fixture
def evaluation_input(request):
    case = evaluation_tests.Phase4FModelEvaluationTests('test_frozen_evaluation_bundle_matches_worker_image_record_contract')
    case.setUp()
    request.addfinalizer(case.tearDown)
    runner = evaluation_tests.EvaluationResultRunner()
    backend = case.service(runner)
    return case, EvaluationInputService(case.store, case.root, backend), backend, runner


def complete_cohorts(case, service):
    classes = case.store.get_project(case.project_id)['classes']
    for split_index, cohort in enumerate(('current_gate', 'core_safety')):
        for index, c in enumerate(classes):
            image = service.upload(case.batch_id, cohort, f'image_{index}.png', png(10 + index + 20 * split_index))
            service.save_label(image['image_id'], [c['class_id']], False)


def test_folder_freeze_uses_existing_backend_and_does_not_launch_gpu(evaluation_input):
    case, service, backend, runner = evaluation_input
    complete_cohorts(case, service)
    snapshot = service.snapshot(case.batch_id)
    assert snapshot['complete']
    result = service.freeze(case.batch_id, confirmed=True, preview_token=snapshot['preview_token'])
    assert result['job']['split_counts'] == {'current_gate': 2, 'core_safety': 2}
    assert runner.spec is None
    assert all(c['status'] == 'pending' for c in backend.list_cohorts(case.project_id))
    assert case.fixture.active_model_id() == 'champion'
    with pytest.raises(PermissionError, match='read-only'):
        service.upload(case.batch_id, 'current_gate', 'late.png', png(90))
    with pytest.raises(PermissionError, match='read-only'):
        service.save_label(snapshot['images'][0]['image_id'], [], True)
    assert case.store.verify_audit_chain()['valid']


def test_folder_round_decision_activates_both_cohorts(evaluation_input):
    case, service, backend, runner = evaluation_input
    complete_cohorts(case, service)
    job = service.freeze(case.batch_id, confirmed=True, preview_token=service.snapshot(case.batch_id)['preview_token'])['job']
    run = backend.wait(backend.start_job(job['job_id'], 'human', 'engineer')['run']['run_id'], timeout=5)
    assert run['status'] == 'result_verified'
    case.store.record_batch_decision(batch_id=case.batch_id, decision='retain', reason='Keep the verified Champion.', actor_type='human', actor_id='engineer', confirmed=True)
    assert all(c['status'] == 'active' for c in backend.list_cohorts(case.project_id))
    assert case.fixture.active_model_id() == 'champion'


def test_incomplete_unconfirmed_and_stale_freezes_are_rejected(evaluation_input):
    case, service, backend, runner = evaluation_input
    with pytest.raises(PermissionError, match='confirmation'):
        service.freeze(case.batch_id, confirmed=False, preview_token='')
    snapshot = service.snapshot(case.batch_id)
    with pytest.raises(ValueError, match='required independent'):
        service.freeze(case.batch_id, confirmed=True, preview_token=snapshot['preview_token'])
    complete_cohorts(case, service)
    with pytest.raises(PermissionError, match='changed'):
        service.freeze(case.batch_id, confirmed=True, preview_token=snapshot['preview_token'])
    assert runner.spec is None


def test_filename_retry_duplicates_and_training_overlap(evaluation_input):
    case, service, backend, runner = evaluation_input
    first = service.upload(case.batch_id, 'current_gate', '100%20.png', png(51))
    assert service.upload(case.batch_id, 'current_gate', '100%20.PNG', png(51))['idempotent']
    assert service.images(case.batch_id)[0]['filename'] == '100%20.png'
    with pytest.raises(ValueError, match='different image'):
        service.upload(case.batch_id, 'current_gate', '100%20.png', png(52))
    with pytest.raises(ValueError, match='duplicated'):
        service.upload(case.batch_id, 'core_safety', 'other.png', png(51))
    with case.store.connect() as db:
        stored = db.execute('SELECT stored_path FROM images LIMIT 1').fetchone()[0]
        # Fixture originals are minimal image probes with valid PNG headers.
        content = Path(stored).read_bytes()
    with pytest.raises(ValueError):
        service.upload(case.batch_id, 'current_gate', 'training.png', content)
    assert len(service.images(case.batch_id)) == 1


def test_table_preview_scope_partial_labels_and_revision(evaluation_input):
    case, service, backend, runner = evaluation_input
    first = service.upload(case.batch_id, 'current_gate', 'one.png', png(61))
    service.upload(case.batch_id, 'current_gate', 'two.png', png(62))
    rows = read_table(service.export(case.batch_id, 'current_gate', 'xlsx', template=True), 'labels.xlsx')
    rows[1][1:] = ['1', '0', '0']
    content = write_table(rows, 'xlsx')
    preview = service.import_table(case.batch_id, 'current_gate', content, 'labels.xlsx')
    assert preview['remaining_unlabeled'] == 1 and not preview['applied']
    with pytest.raises(ValueError, match='missing image'):
        service.import_table(case.batch_id, 'core_safety', content, 'labels.xlsx')
    service.save_label(first['image_id'], [], True)
    with pytest.raises(PermissionError, match='changed'):
        service.import_table(case.batch_id, 'current_gate', content, 'labels.xlsx', confirmed=True, preview_token=preview['preview_token'])
    preview = service.import_table(case.batch_id, 'current_gate', content, 'labels.xlsx')
    service.import_table(case.batch_id, 'current_gate', content, 'labels.xlsx', confirmed=True, preview_token=preview['preview_token'])
    assert not service.snapshot(case.batch_id)['complete']
    counts = WorkflowSnapshotService(case.store).get(case.project_id)['annotation_progress']
    assert counts['kind'] == 'evaluation' and counts['image_count'] == 2 and counts['labeled_count'] == 1
    assert counts['cohorts']['core_safety']['image_count'] == 0


@pytest.mark.parametrize('decision', ['retain', 'promote'])
def test_three_rounds_require_train_and_gate_but_seed_only_once(evaluation_input, decision):
    case, service, backend, runner = evaluation_input
    complete_cohorts(case, service)
    original_champion = case.fixture.active_model_id()
    classes = case.store.get_project(case.project_id)['classes']
    for round_index in range(3):
        if round_index:
            case._prepare_next_round(f'batch_round_{round_index}', f'Round {round_index + 1}')
            snapshot = service.snapshot(case.batch_id)
            assert snapshot['required_cohorts'] == ['current_gate']
            assert snapshot['core_safety_history_count'] == 2 * (round_index + 1)
            with pytest.raises(PermissionError, match='already established'):
                service.upload(case.batch_id, 'core_safety', 'another_seed.png', png(150 + round_index))
            with pytest.raises(ValueError, match='existing Core Safety'):
                service.upload(case.batch_id, 'current_gate', 'old_gate.png', png(10))
            for index, c in enumerate(classes):
                image = service.upload(case.batch_id, 'current_gate', f'gate_{round_index}_{index}.png', png(60 + 10 * round_index + index))
                service.save_label(image['image_id'], [c['class_id']], False)
            counts = WorkflowSnapshotService(case.store).get(case.project_id)['annotation_progress']
            assert set(counts['cohorts']) == {'current_gate'}
        job = service.freeze(case.batch_id, confirmed=True, preview_token=service.snapshot(case.batch_id)['preview_token'])['job']
        assert job['split_counts'] == {'current_gate': 2, 'core_safety': 2 * (round_index + 1)}
        run = backend.wait(backend.start_job(job['job_id'], 'human', 'engineer')['run']['run_id'], timeout=5)
        assert run['status'] == 'result_verified'
        completion = case.store.record_batch_decision(batch_id=case.batch_id, decision=decision, reason='Reviewed independent paired evidence.', actor_type='human', actor_id='engineer', confirmed=True)
        assert completion['round_completion']['cumulative_core_safety_images'] == 2 * (round_index + 2)
        assert len(completion['activated_cohorts']) == (1 if round_index else 2)
        active = backend.list_cohorts(case.project_id, status='active')
        assert sum(c['origin_role'] == 'core_safety_seed' for c in active) == 1
        assert len(active) == round_index + 2
        if decision == 'retain':
            assert case.fixture.active_model_id() == original_champion
        else:
            assert case.fixture.active_model_id() == backend.get_evidence(case.batch_id)['challenger_model_id']
    assert case.store.verify_audit_chain()['valid']


def test_corrupt_disk_data_does_not_publish_partial_package(evaluation_input):
    case, service, backend, runner = evaluation_input
    complete_cohorts(case, service)
    row = service.images(case.batch_id)[0]
    Path(row['stored_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed on disk'):
        service.freeze(case.batch_id, confirmed=True, preview_token=service.snapshot(case.batch_id)['preview_token'])
    assert not list(backend.inbox.glob('folder-evaluation-*.zip'))
    assert not list(backend.inbox.glob('evaluation_draft*.tmp'))


@pytest.mark.parametrize('classes,negative', [([], False), (['unknown'], False), ([], 1)])
def test_invalid_manual_labels_do_not_apply(evaluation_input, classes, negative):
    case, service, backend, runner = evaluation_input
    image = service.upload(case.batch_id, 'current_gate', 'one.png', png(71))
    with pytest.raises(ValueError):
        service.save_label(image['image_id'], classes, negative)
    assert service.images(case.batch_id)[0]['annotation_status'] == 'unlabeled'


def test_legacy_zip_gate_cannot_be_reimported_as_next_round_train(evaluation_input):
    case, service, backend, runner = evaluation_input
    job = backend.create_job(case.batch_id, case.snapshot.name, 'human', 'engineer')['job']
    _, gate_content = case._first_job_image(job)
    run = backend.wait(backend.start_job(job['job_id'], 'human', 'engineer')['run']['run_id'], timeout=5)
    assert run['status'] == 'result_verified'
    case.store.record_batch_decision(batch_id=case.batch_id, decision='retain', reason='Keep the verified Champion.', actor_type='human', actor_id='engineer', confirmed=True)
    next_batch = case.store.create_maintenance_batch(project_id=case.project_id, batch_name='Next Train')['maintenance_batch']
    dataset = case.store.create_maintenance_dataset(batch_id=next_batch['batch_id'], name='Train')['dataset']

    def register(content, filename):
        path = case.root / filename
        path.write_bytes(content)
        return case.store.register_image(dataset_id=dataset['dataset_id'], filename=filename, stored_path=str(path), sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content), mime_type='image/png', width=1, height=1, health_status='ok')

    # A genuinely fresh Train image is accepted, including after inventory caching.
    register(png(233), 'fresh.png')
    with pytest.raises(ValueError, match='reserved for independent evaluation'):
        register(gate_content, 'renamed_gate.png')
    assert len(case.store.list_images(dataset['dataset_id'])) == 1
    # Missing history must fail closed rather than silently permit leakage.
    artifact = Path(job['bundle_path'])
    artifact.chmod(0o666)  # This disposable fixture intentionally simulates lost history.
    artifact.unlink()
    with pytest.raises(ValueError, match='restore its unchanged artifacts'):
        register(png(234), 'fresh_after_missing_history.png')
