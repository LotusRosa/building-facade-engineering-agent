"""Exercise the real HTTP handler without starting production runtime services."""
from __future__ import annotations

import ast
import json
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from facade_agent.application.annotation_tables import AnnotationTableService, read_table, write_table
from facade_agent.storage import Store
from facade_agent.adapters.llm import LLMManager
from facade_agent.application.agent_runtime import AgentRuntime
from facade_agent.application.workflow import WorkflowSnapshotService
from facade_agent.tools import build_phase1_registry
from test_evaluation_folder_input import png, evaluation_input


@pytest.fixture
def endpoint(request):
    temporary = tempfile.TemporaryDirectory()
    request.addfinalizer(temporary.cleanup)
    root = Path(temporary.name)
    store = Store(root / 'agent.sqlite3')
    project = store.create_project(project_name='HTTP labels', class_names=['crack', 'spalling'])['project']
    store.transition_project(project_id=project['project_id'], action='confirm_taxonomy', confirmed=True)
    dataset = store.create_dataset(project_id=project['project_id'], name='folder', role='initial_training')['dataset']
    source = Path(__file__).resolve().parents[1] / 'facade_agent' / 'server.py'
    tree = ast.parse(source.read_text(encoding='utf-8'), str(source))
    # Imports + actual Handler, excluding production global service initialization.
    selected = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) or isinstance(n, ast.ClassDef) and n.name == 'Handler']
    namespace = {'__name__': 'facade_agent._handler_test', '__package__': 'facade_agent'}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    namespace.update(STORE=store, ROOT=root, PROJECT_FILES=root / 'projects', ANNOTATION_TABLES=AnnotationTableService(store))
    server = ThreadingHTTPServer(('127.0.0.1', 0), namespace['Handler'])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def stop():
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    request.addfinalizer(stop)

    def send(path, body=None, params=None):
        url = f'http://127.0.0.1:{server.server_address[1]}{path}'
        if params:
            url += '?' + urlencode(params)
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        try:
            response = urlopen(Request(url, data=body), timeout=5)
        except HTTPError as exc:
            response = exc
        with response:
            content = response.read()
            if 'application/json' in response.headers.get('Content-Type', ''):
                content = json.loads(content)
            return response.status, content
    send.namespace = namespace
    return store, project, dataset, send


def test_chat_http_shares_workflow_language_and_requires_boolean_approval(endpoint):
    store, _, _, send = endpoint
    project = store.create_project(project_name='Chat API', class_names=['crack'])['project']
    workflow = WorkflowSnapshotService(store)
    registry = build_phase1_registry(store)
    send.namespace.update(AGENT=AgentRuntime(store, registry, LLMManager(), workflow), TOOLS=registry, WORKFLOW=workflow)
    status, response = send('/api/agent/chat', {'text': '继续', 'project_id': project['project_id'], 'language': 'en'})
    assert status == 200 and 'Confirm frozen classes' in response['result']['message']
    pending = response['result']['pending']['pending_id']
    status, data = send('/api/agent/confirm', {'pending_id': pending, 'approved': 'false', 'language': 'en'})
    assert status == 400 and not data['ok']
    assert store.get_pending_tool_call(pending)['status'] == 'pending'
    status, data = send('/api/tool/run', {'tool': 'confirm_taxonomy', 'arguments': {'project_id': project['project_id']}, 'confirmed': 'false'})
    assert status == 400 and not data['ok']
    assert store.get_project(project['project_id'])['state'] == 'PROJECT_CREATED'
    status, data = send('/api/agent/confirm', {'pending_id': pending, 'approved': True, 'language': 'en'})
    assert status == 200 and 'Engineer confirmed' in data['result']['message']
    assert store.get_project(project['project_id'])['state'] == 'TAXONOMY_CONFIRMED'
    status, response = send('/api/agent/chat', {'text': 'continue', 'project_id': project['project_id'], 'language': 'zh'})
    assert status == 200 and '导入并标注初始训练数据' in response['result']['message']
    assert store.verify_audit_chain()['valid']


def test_http_upload_export_preview_confirm_and_readonly(endpoint):
    store, project, dataset, send = endpoint
    filename = 'literal%20中文.png'
    params = {'dataset_id': dataset['dataset_id'], 'filename': filename}
    status, result = send('/api/datasets/upload', png(100), params)
    assert status == 201 and result['image']['filename'] == filename
    status, result = send('/api/datasets/upload', png(100), params)
    assert status == 200 and result['idempotent']
    status, result = send('/api/datasets/upload', png(101), params)
    assert status == 400 and not result['ok']
    assert len(store.list_images(dataset['dataset_id'])) == 1
    status, content = send('/api/annotations/table', params={'project_id': project['project_id'], 'dataset_id': dataset['dataset_id'], 'template': 'true'})
    rows = read_table(content, 'labels.xlsx')
    assert rows[1] == [filename, '', '', '']
    rows[1][1:] = ['1', '1', '0']
    body = write_table(rows, 'xlsx')
    params = {'dataset_id': dataset['dataset_id'], 'filename': 'labels.xlsx'}
    status, preview = send('/api/annotations/table/import', body, params)
    assert status == 200 and not preview['result']['applied']
    status, failure = send('/api/annotations/table/import', body, params | {'confirmed': 'true'})
    assert status == 400 and not failure['ok']
    status, applied = send('/api/annotations/table/import', body, params | {'confirmed': 'true', 'preview_token': preview['result']['preview_token']})
    assert status == 200 and applied['result']['validation']['valid']
    status, failure = send('/api/datasets/upload', png(102), params={'dataset_id': dataset['dataset_id'], 'filename': 'late.png'})
    assert status == 400


def test_bad_table_is_atomic_and_boolean_labels_are_not_coerced(endpoint):
    store, project, dataset, send = endpoint
    images = []
    for index in range(2):
        _, data = send('/api/datasets/upload', png(index + 110), {'dataset_id': dataset['dataset_id'], 'filename': f'{index}.png'})
        images.append(data['image'])
    rows = [['filename', 'crack', 'spalling', 'no_defect'], ['0.png', 1, 0, 0], ['1.png', 0, 0, 0]]
    status, result = send('/api/annotations/table/import', write_table(rows, 'csv'), {'dataset_id': dataset['dataset_id'], 'filename': 'labels.csv'})
    assert status == 400 and 'Row 3' in result['error']
    assert all(i['annotation_status'] == 'unlabeled' for i in store.list_images(dataset['dataset_id']))
    status, result = send('/api/annotations/save', {'image_id': images[0]['image_id'], 'class_ids': [], 'no_defect': 'false'})
    assert status == 400 and 'boolean' in result['error']


def test_export_rejects_other_project_and_invalid_format(endpoint):
    store, project, dataset, send = endpoint
    other = store.create_project(project_name='Other', class_names=['other'])['project']
    status, result = send('/api/annotations/table', params={'project_id': other['project_id'], 'dataset_id': dataset['dataset_id']})
    assert status == 400 and 'another project' in result['error']
    status, result = send('/api/annotations/table', params={'project_id': project['project_id'], 'format': 'xls'})
    assert status == 400


def test_evaluation_folder_routes_require_confirmation_and_freeze_without_gpu(endpoint, evaluation_input):
    _, _, _, send = endpoint
    case, service, backend, runner = evaluation_input
    send.namespace.update(STORE=case.store, ROOT=case.root, PROJECT_FILES=case.root / 'projects', EVALUATION_INPUT=service)
    classes = case.store.get_project(case.project_id)['classes']
    for cohort_index, cohort in enumerate(('current_gate', 'core_safety')):
        for index, c in enumerate(classes):
            params = {'batch_id': case.batch_id, 'cohort': cohort, 'filename': f'{index}.png'}
            content = png(180 + cohort_index * 10 + index)
            status, data = send('/api/evaluation/input/upload', content, params)
            assert status == 400 and not data['ok']
            status, data = send('/api/evaluation/input/upload', content, params | {'confirmed': 'true'})
            assert status == 200
            image_id = data['result']['image_id']
            status, image = send('/api/evaluation/input/image', params={'image_id': image_id})
            assert status == 200 and image == content
            status, data = send('/api/evaluation/input/save', {'image_id': image_id, 'class_ids': [c['class_id']], 'no_defect': False})
            assert status == 200 and data['result']['saved']
    status, table = send('/api/evaluation/input/table', params={'batch_id': case.batch_id, 'cohort': 'current_gate', 'format': 'xlsx'})
    assert status == 200 and len(read_table(table, 'labels.xlsx')) == 3
    status, data = send('/api/evaluation/input/status', params={'batch_id': case.batch_id})
    assert status == 200 and data['result']['complete']
    params = {'batch_id': case.batch_id, 'preview_token': data['result']['preview_token']}
    status, data = send('/api/evaluation/input/freeze', params | {'confirmed': 'true'})
    assert status == 400 and not data['ok']
    status, data = send('/api/evaluation/input/freeze', params | {'confirmed': True})
    assert status == 200 and data['result']['job']['split_counts'] == {'current_gate': 2, 'core_safety': 2}
    assert runner.spec is None
    status, data = send('/api/evaluation/input/upload', png(200), {'batch_id': case.batch_id, 'cohort': 'current_gate', 'filename': 'late.png', 'confirmed': 'true'})
    assert status == 400 and 'read-only' in data['error']
