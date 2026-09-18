import copy
import random

import pytest

from app.models import Interpretation, Plan, Scenario
from app.optimizer import InfeasiblePlan, _optimize_model as optimize
from app.validation import replay
from tests.conftest import CASES


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_public_optimum_and_reference_replay(case):
    scenario = Scenario.model_validate(case["input"])
    interpretation = Interpretation(directive_interpretation=case["expected_output"]["directive_interpretation"])
    replay(scenario, Plan.model_validate(case["expected_output"]))
    plan = optimize(scenario, interpretation)
    replay(scenario, plan)
    assert plan.total_cost_bdt == pytest.approx(case["expected_output"]["total_cost_bdt"], abs=1e-5)


def no_op():
    return Interpretation(directive_interpretation=[dict(
        note_index=0, applies=False, directive_type="no_op", structured_adjustment=None,
        explanation="Unrelated note.")])


def small_scenario(seed):
    rng = random.Random(seed)
    capacity = rng.randint(0, 10)
    reserve = rng.randint(0, capacity)
    initial = rng.randint(reserve, capacity)
    return Scenario.model_validate(dict(
        scenario_id=f"random-{seed}", operator_notes=["The office calendar was updated."],
        battery=dict(capacity_kwh=capacity, minimum_energy_kwh=reserve,
                     initial_energy_kwh=initial, max_charge_kwh_per_hour=rng.randint(0, 5),
                     max_discharge_kwh_per_hour=rng.randint(0, 5)),
        hours=[dict(hour=h, demand_kwh=rng.randint(0, 12), solar_kwh=rng.randint(0, 14),
                    tariff_bdt_per_kwh=rng.randint(0, 20)) for h in range(24)],
    ))


def discrete_oracle(scenario):
    # Exhaustive DP over integer battery states, independent of the LP formulation.
    # Integer network-flow input has an integral optimum in this lossless model.
    b = scenario.battery
    states = {int(b.initial_energy_kwh): 0.0}
    for hour in scenario.hours:
        next_states = {}
        for before, cost in states.items():
            for after in range(int(b.minimum_energy_kwh), int(b.capacity_kwh) + 1):
                delta = after - before
                if not -b.max_discharge_kwh_per_hour <= delta <= b.max_charge_kwh_per_hour:
                    continue
                if hour.demand_kwh + delta < 0:
                    continue
                grid = max(0, hour.demand_kwh + delta - hour.solar_kwh)
                value = cost + grid * hour.tariff_bdt_per_kwh
                next_states[after] = min(next_states.get(after, float("inf")), value)
        states = next_states
    return states[int(b.initial_energy_kwh)]


@pytest.mark.parametrize("seed", range(100))
def test_random_optimum_against_independent_dynamic_program(seed):
    scenario = small_scenario(seed)
    plan = optimize(scenario, no_op())
    assert plan.total_cost_bdt == pytest.approx(discrete_oracle(scenario), abs=1e-5)


def test_fractional_values_and_unsorted_hours(case):
    data = case["input"]
    for hour in data["hours"]:
        hour["demand_kwh"] *= 0.1234567
        hour["solar_kwh"] *= 0.1234567
        hour["tariff_bdt_per_kwh"] *= 1.234567
    for field in data["battery"]:
        data["battery"][field] *= 0.1234567
    data["hours"].reverse()
    scenario = Scenario.model_validate(data)
    directives = Interpretation(directive_interpretation=case["expected_output"]["directive_interpretation"])
    plan = optimize(scenario, directives)
    replay(scenario, plan)
    assert plan.total_cost_bdt == pytest.approx(38365 * 0.1234567 * 1.234567, abs=1e-5)


def test_infeasible_directives_are_rejected(case):
    data = case["input"]
    data["operator_notes"] = ["No grid all day."]
    scenario = Scenario.model_validate(data)
    interpretation = Interpretation(directive_interpretation=[dict(
        note_index=0, applies=True, directive_type="max_grid_window",
        structured_adjustment=dict(hours=list(range(24)), max_grid_kwh=0), explanation="No grid.")])
    with pytest.raises(InfeasiblePlan):
        optimize(scenario, interpretation)


@pytest.mark.parametrize("mutation", ["balance", "neutrality", "total", "directive"])
def test_replay_rejects_corrupted_plans(case, mutation):
    scenario = Scenario.model_validate(case["input"])
    plan = copy.deepcopy(case["expected_output"])
    if mutation == "balance":
        plan["hourly_plan"][0]["grid_kwh"] += 1
    elif mutation == "neutrality":
        plan["hourly_plan"][-1]["battery_energy_after_kwh"] += 1
    elif mutation == "total":
        plan["total_cost_bdt"] += 1
    else:
        plan["directive_interpretation"][0]["structured_adjustment"]["factor"] = 0
    with pytest.raises(ValueError):
        replay(scenario, Plan.model_validate(plan))
