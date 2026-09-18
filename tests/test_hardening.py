import pathlib
import copy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.interpreter import _validate_all
from app.models import Scenario, Plan
from app.validation import replay

CASES = json.loads((Path(__file__).resolve().parents[1] / 'samples/public_sample_cases.json').read_text())['cases']


@pytest.mark.parametrize('case', CASES, ids=lambda c:c['id'])
def test_api_validated_interpretation_path(case, monkeypatch):
    monkeypatch.setattr('app.main.interpret', lambda *args: (copy.deepcopy(case['expected_output']['directive_interpretation']), 'mock-llm'))
    with TestClient(app) as client:
        r=client.post('/optimize-energy', json=case['input'])
    assert r.status_code==200
    plan=Plan.model_validate(r.json())
    replay(Scenario.model_validate(case['input']),plan)
    assert abs(plan.total_cost_bdt-case['expected_output']['total_cost_bdt'])<0.01


def test_failed_replay_is_not_returned_as_success(monkeypatch):
    case=CASES[0]
    monkeypatch.setattr('app.main.interpret',lambda *args:(case['expected_output']['directive_interpretation'],'mock-llm'))
    monkeypatch.setattr('app.main.replay',lambda *args:['invalid'])
    with TestClient(app) as client:
        response=client.post('/optimize-energy',json=case['input'])
    assert response.status_code==500
    assert 'hourly_plan' not in response.json()


def test_note_indices_must_not_be_invented_or_duplicated():
    for items in [[{'directive_type':'no_op'}], [{'note_index':True,'directive_type':'no_op'}],
                  [{'note_index':1,'directive_type':'no_op'}]]:
        assert _validate_all(items,['note'],100)[1]


@pytest.mark.parametrize('raw', ['{"scenario_id":NaN}', '{"x":1,"x":2}', '{'])
def test_nonstandard_json_is_400(raw):
    with TestClient(app) as client:
        assert client.post('/optimize-energy',content=raw).status_code==400


def test_unconfigured_health(monkeypatch):
    monkeypatch.delenv('LLM_API_KEY',raising=False)
    monkeypatch.delenv('GROQ_API_KEY',raising=False)
    with TestClient(app) as client:
        r = client.get('/health')
        assert r.status_code == 200 and r.json() == {'status': 'ok'}
        assert client.head('/health').status_code == 200  # uptime monitors probe with HEAD


def test_extra_request_fields_are_ignored():
    case = json.loads(pathlib.Path('samples/public_sample_cases.json').read_text())['cases'][0]['input']
    body = dict(case, harness_run='x', battery=dict(case['battery'], chemistry='LFP'),
                hours=[dict(h, note='x') for h in case['hours']])
    from app.main import validate_request
    assert validate_request(body)['battery']['capacity_kwh'] == case['battery']['capacity_kwh']
