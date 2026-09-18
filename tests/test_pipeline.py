"""Offline tests (no LLM key needed): optimizer optimality, guardrails, fallback parser, API contract."""
import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from app import llm
from app.fallback import interpret_note_fallback
from app.guardrails import GuardrailError, validate_entry
from app.main import app
from app.optimizer import optimize
from app.replay import replay

CASES = json.loads((pathlib.Path(__file__).resolve().parents[1] / "samples/public_sample_cases.json").read_text())["cases"]
@pytest.fixture
def client():
    from app.interpreter import _CACHE
    _CACHE.clear()
    with TestClient(app) as instance:
        yield instance



@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_optimizer_reaches_reference_cost(case):
    req, exp = case["input"], case["expected_output"]
    d = exp["directive_interpretation"]
    r = optimize(req["hours"], req["battery"], d)
    assert replay(req, dict(r, directive_interpretation=d)) == []
    assert abs(r["total_cost_bdt"] - exp["total_cost_bdt"]) <= 0.01


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_fallback_parser_matches_samples(case):
    cap = case["input"]["battery"]["capacity_kwh"]
    for i, (note, exp) in enumerate(zip(case["input"]["operator_notes"], case["expected_output"]["directive_interpretation"])):
        got = validate_entry(interpret_note_fallback(note), i, cap)
        assert got["directive_type"] == exp["directive_type"]
        assert got["structured_adjustment"] == exp["structured_adjustment"]


@pytest.mark.parametrize("note,hours,factor", [
    ("PV production will drop to about 20% between 13:00 and 15:00.", [13, 14], 0.2),
    ("Panel washing from one until three will leave roughly one-fifth of normal solar output.", [13, 14], 0.2),
    ("Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window.", [13, 14], 0.2),
])
def test_fallback_paraphrases(note, hours, factor):
    got = validate_entry(interpret_note_fallback(note), 0, 200)
    assert got["structured_adjustment"] == {"hours": hours, "factor": factor}


def test_guardrails():
    ok = validate_entry({"directive_type": "no_charge_window", "windows": [[22, 2]]}, 0, 100)
    assert ok["structured_adjustment"]["hours"] == [0, 1, 22, 23] and ok["applies"] is True
    pct = validate_entry({"directive_type": "minimum_battery_reserve", "windows": [[18, 21]], "minimum_percent_of_capacity": 50}, 0, 200)
    assert pct["structured_adjustment"]["minimum_energy_kwh"] == 100
    noop = validate_entry({"directive_type": "no_op", "windows": [[1, 2]]}, 1, 100)
    assert noop == {**noop, "applies": False, "structured_adjustment": None}
    for bad in (
        {"directive_type": "demand_increase", "windows": [[1, 2]]},
        {"directive_type": "solar_reduction", "windows": [[1, 2]], "factor": 1.5e3},
        {"directive_type": "solar_reduction", "windows": [[1, 2]], "factor": 1.5},
        {"directive_type": "minimum_battery_reserve", "windows": [[1, 2]], "minimum_energy_kwh": 999},
        {"directive_type": "max_grid_window", "windows": [[1, 25]], "max_grid_kwh": 10},
        {"directive_type": "max_grid_window", "windows": [[1, 3]], "max_grid_kwh": -1},
    ):
        with pytest.raises(GuardrailError):
            validate_entry(bad, 0, 100)


def test_health(client, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    assert client.get("/health").json() == {"status": "ok"}


def test_malformed_requests(client):
    assert client.post("/optimize-energy", content=b"{not json").status_code == 400
    assert client.post("/optimize-energy", json={"scenario_id": "x"}).status_code == 400
    req = json.loads(json.dumps(CASES[0]["input"]))
    req["hours"] = req["hours"][:23]
    assert client.post("/optimize-energy", json=req).status_code == 400


def test_end_to_end_without_llm(monkeypatch, client):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    for case in CASES:
        r = client.post("/optimize-energy", json=case["input"])
        assert r.status_code == 500
        assert "hourly_plan" not in r.json()


def test_llm_bad_output_is_contained(monkeypatch, client):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr("app.interpreter.interpret_notes_llm",
                        lambda notes, cap, feedback=None, **kwargs: [{"note_index": 0, "directive_type": "delete_campus"}])
    case = CASES[1]
    r = client.post("/optimize-energy", json=dict(case["input"], scenario_id="LLM-BAD"))
    assert r.status_code == 500
    assert "delete_campus" not in r.text
