"""Independent replay checker, mirroring the judge rules (Problem Statement sections 9 and 11)."""
from __future__ import annotations

from .optimizer import H, build_constraints

TOL = 0.01


def replay(request: dict, response: dict, directives: list[dict] | None = None) -> list[str]:
    """Return a list of rule violations (empty list == valid plan)."""
    errors: list[str] = []
    hours, battery = request["hours"], request["battery"]
    directives = response["directive_interpretation"] if directives is None else directives
    cons = build_constraints(hours, battery, directives)
    plan = response["hourly_plan"]

    if [p["hour"] for p in plan] != list(range(H)):
        return ["hourly_plan must contain hours 0..23 in order"]

    e = float(battery["initial_energy_kwh"])
    for p in plan:
        h = p["hour"]
        for k in ("grid_kwh", "solar_used_kwh", "battery_kwh", "battery_energy_after_kwh"):
            if p[k] < -TOL:
                errors.append(f"h{h}: negative {k}")
        act, b = p["battery_action"], p["battery_kwh"]
        ch = b if act == "charge" else 0.0
        dis = b if act == "discharge" else 0.0
        if act == "idle" and abs(b) > TOL:
            errors.append(f"h{h}: idle with battery_kwh={b}")
        if act not in ("charge", "discharge", "idle"):
            errors.append(f"h{h}: bad action {act}")
        if ch > battery["max_charge_kwh_per_hour"] + TOL:
            errors.append(f"h{h}: charge rate exceeded")
        if dis > battery["max_discharge_kwh_per_hour"] + TOL:
            errors.append(f"h{h}: discharge rate exceeded")
        if ch > TOL and not cons.charge_allowed[h]:
            errors.append(f"h{h}: charging inside no_charge_window")
        if dis > TOL and not cons.discharge_allowed[h]:
            errors.append(f"h{h}: discharging inside no_discharge_window")
        if p["solar_used_kwh"] > cons.effective_solar[h] + TOL:
            errors.append(f"h{h}: solar overuse")
        bal = p["grid_kwh"] + p["solar_used_kwh"] + dis - hours[h]["demand_kwh"] - ch
        if abs(bal) > TOL:
            errors.append(f"h{h}: energy balance off by {bal:.4f}")
        e = e + ch - dis
        if abs(e - p["battery_energy_after_kwh"]) > TOL:
            errors.append(f"h{h}: battery transition mismatch")
        if e < cons.reserve[h] - TOL:
            errors.append(f"h{h}: battery below reserve ({e:.2f} < {cons.reserve[h]})")
        if e > battery["capacity_kwh"] + TOL:
            errors.append(f"h{h}: battery above capacity")
        cap = cons.grid_cap[h]
        if cap is not None and p["grid_kwh"] > cap + TOL:
            errors.append(f"h{h}: grid cap exceeded ({p['grid_kwh']} > {cap})")

    if abs(e - battery["initial_energy_kwh"]) > TOL:
        errors.append("end-of-day battery neutrality violated")
    tot = sum(p["grid_kwh"] for p in plan)
    cost = sum(p["grid_kwh"] * hours[p["hour"]]["tariff_bdt_per_kwh"] for p in plan)
    peak = max(p["grid_kwh"] for p in plan)
    if abs(tot - response["total_grid_kwh"]) > TOL:
        errors.append("total_grid_kwh mismatch")
    if abs(cost - response["total_cost_bdt"]) > TOL:
        errors.append("total_cost_bdt mismatch")
    if abs(peak - response["peak_grid_kwh"]) > TOL:
        errors.append("peak_grid_kwh mismatch")
    return errors
