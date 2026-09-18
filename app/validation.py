"""Independent schedule replay; no optimizer matrices or precomputed limits reused."""
from math import fsum, isclose

from app.models import Interpretation, Plan, Scenario


def replay(scenario: Scenario, plan: Plan, tolerance: float = 1e-5) -> None:
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def equal(a, b):
        return isclose(a, b, abs_tol=tolerance, rel_tol=0)

    require(plan.scenario_id == scenario.scenario_id, "scenario_id mismatch")
    Interpretation(directive_interpretation=plan.directive_interpretation).validate_for(scenario)
    require([p.hour for p in plan.hourly_plan] == list(range(24)), "Invalid hourly plan order")
    b = scenario.battery
    energy = b.initial_energy_kwh
    for hour, row in zip(scenario.hours, plan.hourly_plan):
        solar, reserve = hour.solar_kwh, b.minimum_energy_kwh
        charge_limit, discharge_limit = b.max_charge_kwh_per_hour, b.max_discharge_kwh_per_hour
        grid_limit = float("inf")
        for directive in plan.directive_interpretation:
            a = directive.structured_adjustment
            if a is None or hour.hour not in a.hours:
                continue
            match directive.directive_type:
                case "solar_reduction": solar = min(solar, hour.solar_kwh * a.factor)
                case "minimum_battery_reserve": reserve = max(reserve, a.minimum_energy_kwh)
                case "no_charge_window": charge_limit = 0.0
                case "no_discharge_window": discharge_limit = 0.0
                case "max_grid_window": grid_limit = min(grid_limit, a.max_grid_kwh)
        charge = row.battery_kwh if row.battery_action == "charge" else 0.0
        discharge = row.battery_kwh if row.battery_action == "discharge" else 0.0
        require(row.battery_action != "idle" or row.battery_kwh == 0, "Nonzero idle action")
        require(charge <= charge_limit + tolerance, "Charge limit exceeded")
        require(discharge <= discharge_limit + tolerance, "Discharge limit exceeded")
        require(row.grid_kwh <= grid_limit + tolerance, "Grid cap exceeded")
        require(row.solar_used_kwh <= solar + tolerance, "Solar availability exceeded")
        require(equal(row.grid_kwh + row.solar_used_kwh + discharge, hour.demand_kwh + charge),
                "Hourly energy balance failed")
        energy += charge - discharge
        require(equal(energy, row.battery_energy_after_kwh), "Battery state transition failed")
        require(reserve - tolerance <= energy <= b.capacity_kwh + tolerance, "Battery bound failed")
    require(equal(energy, b.initial_energy_kwh), "End-of-day neutrality failed")
    require(equal(plan.total_grid_kwh, fsum(p.grid_kwh for p in plan.hourly_plan)), "Grid total mismatch")
    require(equal(plan.peak_grid_kwh, max(p.grid_kwh for p in plan.hourly_plan)), "Peak mismatch")
    require(equal(plan.total_cost_bdt, fsum(p.grid_kwh * h.tariff_bdt_per_kwh
                for p, h in zip(plan.hourly_plan, scenario.hours))), "Cost mismatch")
