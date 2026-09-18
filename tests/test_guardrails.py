import copy

import pytest

from app.models import Interpretation, Scenario
from app.main import _strict_json as strict_json


@pytest.mark.parametrize("field,value", [
    ("hours", [13, 13]), ("hours", [14, 13]), ("hours", [-1]), ("hours", [24]),
    ("hours", [13.0]), ("hours", [True]), ("hours", []),
    ("factor", -0.1), ("factor", 1.1), ("factor", float("nan")),
    ("factor", float("inf")), ("factor", "0.2"), ("factor", True),
    ("demand_kwh", 30),
])
def test_bad_model_adjustments_rejected(case, field, value):
    directives = copy.deepcopy(case["expected_output"]["directive_interpretation"])
    directives[0]["structured_adjustment"][field] = value
    with pytest.raises(ValueError):
        Interpretation(directive_interpretation=directives).validate_for(Scenario.model_validate(case["input"]))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "order", "applies", "no_op", "type", "capacity"])
def test_note_mapping_and_semantics(case, mutation):
    directives = copy.deepcopy(case["expected_output"]["directive_interpretation"])
    if mutation == "missing": directives.pop()
    elif mutation == "duplicate": directives[1]["note_index"] = 0
    elif mutation == "order": directives.reverse()
    elif mutation == "applies": directives[0]["applies"] = False
    elif mutation == "no_op": directives[1]["structured_adjustment"] = {"hours": [1]}
    elif mutation == "type": directives[0]["directive_type"] = "change_tariff"
    else:
        directives[0]["directive_type"] = "minimum_battery_reserve"
        directives[0]["structured_adjustment"] = {"hours": [18], "minimum_energy_kwh": 100000}
    with pytest.raises(ValueError):
        Interpretation(directive_interpretation=directives).validate_for(Scenario.model_validate(case["input"]))


@pytest.mark.parametrize("text", ['{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'])
def test_nonstandard_json_rejected(text):
    with pytest.raises(ValueError):
        strict_json(text)
