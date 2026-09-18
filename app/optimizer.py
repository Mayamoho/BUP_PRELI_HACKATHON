"""96-variable exact LP. A signed battery flow prevents simultaneous actions."""
from dataclasses import dataclass
from math import fsum
from threading import Lock

_SOLVER_LOCK = Lock()

import numpy as np
from scipy.optimize import linprog

from app.models import Interpretation, Plan, PlanHour, Scenario


class InfeasiblePlan(ValueError):
    pass


@dataclass
class Limits:
    solar: list[float]
    reserve: list[float]
    charge: list[float]
    discharge: list[float]
    grid: list[float]


def apply_directives(scenario: Scenario, interpretation: Interpretation) -> Limits:
    interpretation.validate_for(scenario)
    b = scenario.battery
    limits = Limits(
        [h.solar_kwh for h in scenario.hours],
        [b.minimum_energy_kwh] * 24,
        [b.max_charge_kwh_per_hour] * 24,
        [b.max_discharge_kwh_per_hour] * 24,
        [h.demand_kwh + b.max_charge_kwh_per_hour for h in scenario.hours],
    )
    for d in interpretation.directive_interpretation:
        a = d.structured_adjustment
        if a is None:
            continue
        for h in a.hours:
            if d.directive_type == "solar_reduction":
                # Overlapping absolute forecast fractions: use the tightest cap.
                limits.solar[h] = min(limits.solar[h], scenario.hours[h].solar_kwh * a.factor)
            elif d.directive_type == "minimum_battery_reserve":
                limits.reserve[h] = max(limits.reserve[h], a.minimum_energy_kwh)
            elif d.directive_type == "no_charge_window":
                limits.charge[h] = 0.0
            elif d.directive_type == "no_discharge_window":
                limits.discharge[h] = 0.0
            elif d.directive_type == "max_grid_window":
                limits.grid[h] = min(limits.grid[h], a.max_grid_kwh)
    return limits


def _optimize_model(scenario: Scenario, interpretation: Interpretation) -> Plan:
    limits = apply_directives(scenario, interpretation)
    battery = scenario.battery
    # x = [grid(24), solar_used(24), signed_charge(24), energy_after(24)]
    cost = np.zeros(96)
    cost[:24] = [h.tariff_bdt_per_kwh for h in scenario.hours]
    a_eq = np.zeros((49, 96))
    b_eq = np.zeros(49)
    for h, hour in enumerate(scenario.hours):
        a_eq[h, h] = a_eq[h, 24 + h] = 1
        a_eq[h, 48 + h] = -1
        b_eq[h] = hour.demand_kwh
        a_eq[24 + h, 72 + h] = 1
        a_eq[24 + h, 48 + h] = -1
        if h:
            a_eq[24 + h, 72 + h - 1] = -1
        else:
            b_eq[24 + h] = battery.initial_energy_kwh
    a_eq[48, 95] = 1
    b_eq[48] = battery.initial_energy_kwh
    bounds = (
        [(0, v) for v in limits.grid]
        + [(0, v) for v in limits.solar]
        + [(-limits.discharge[h], limits.charge[h]) for h in range(24)]
        + [(v, battery.capacity_kwh) for v in limits.reserve]
    )
    with _SOLVER_LOCK:
        result = linprog(
            cost, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs",
            options={"time_limit": 3.0, "primal_feasibility_tolerance": 1e-9,
                     "dual_feasibility_tolerance": 1e-9},
        )
    if result.status == 2:
        raise InfeasiblePlan("No feasible schedule under the interpreted directives")
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("Optimizer did not produce a verified optimum")
    rows = []
    energy = battery.initial_energy_kwh
    for h in range(24):
        delta = float(result.x[48 + h])
        if abs(delta) < 1e-9:
            delta = 0.0
        energy += delta
        rows.append(PlanHour(
            hour=h, grid_kwh=max(0.0, float(result.x[h])),
            solar_used_kwh=max(0.0, float(result.x[24 + h])),
            battery_action="charge" if delta > 0 else "discharge" if delta < 0 else "idle",
            battery_kwh=abs(delta), battery_energy_after_kwh=max(0.0, energy),
        ))
    active = sum(d.applies for d in interpretation.directive_interpretation)
    plan = Plan(
        scenario_id=scenario.scenario_id,
        directive_interpretation=interpretation.directive_interpretation,
        hourly_plan=rows,
        total_grid_kwh=fsum(r.grid_kwh for r in rows),
        total_cost_bdt=fsum(r.grid_kwh * scenario.hours[r.hour].tariff_bdt_per_kwh for r in rows),
        peak_grid_kwh=max(r.grid_kwh for r in rows),
        plan_summary=(f"Minimum-cost schedule respecting {active} applicable operator directive(s). "
                      "Solar and battery shifting reduce grid cost; the battery finishes at its initial energy. "
                      "All hourly constraints and totals were independently replayed."),
    )
    # Import here to keep replay independent of the LP implementation.
    from app.validation import replay
    replay(scenario, plan)
    return plan


H = 24

def optimize(hours: list[dict], battery: dict, directives: list[dict]) -> dict:
    scenario = Scenario.model_validate(dict(
        scenario_id="optimizer", operator_notes=["Interpreted note"] * len(directives),
        hours=hours, battery=battery))
    interpretation = Interpretation(directive_interpretation=directives)
    plan = _optimize_model(scenario, interpretation)
    result = plan.model_dump()
    result.pop("scenario_id")
    return {**result, "feasible": True, "slack_used": False}
