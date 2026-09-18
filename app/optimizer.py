"""24-hour battery/grid scheduling as a linear program (SciPy HiGHS).

Decision variables per hour h (all >= 0):
    g[h]  grid import            s[h]  solar used (<= effective solar)
    c[h]  battery charge          d[h]  battery discharge
    e[h]  battery energy after hour h
    rs[h] reserve slack           gs[h] grid-cap slack   (only penalised escape hatches)

Hard constraints:
    g + s + d - c = demand                         (energy balance)
    e[h] = e[h-1] + c[h] - d[h],  e[-1] = initial  (state transition)
    minimum <= e[h] <= capacity                    (bounds)
    c <= max_charge, d <= max_discharge            (rate limits; 0 inside no-charge / no-discharge windows)
    e[23] = initial                                (end-of-day neutrality)
Directive constraints (slack is priced at BIG_M so it is only used if the scenario is infeasible):
    e[h] + rs[h] >= reserve[h]
    g[h] - gs[h] <= grid_cap[h]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

H = 24
BIG_M = 1e6
CYCLE_EPS = 1e-6  # tiny tie-breaker: avoid pointless charge/discharge cycling
ROUND = 4


@dataclass
class Constraints:
    """Per-hour effective limits after directives are applied."""

    effective_solar: list[float]
    reserve: list[float]  # effective minimum energy per hour
    charge_allowed: list[bool]
    discharge_allowed: list[bool]
    grid_cap: list[float | None]


def build_constraints(hours: list[dict], battery: dict, directives: list[dict]) -> Constraints:
    solar = [float(h["solar_kwh"]) for h in hours]
    base_min = float(battery["minimum_energy_kwh"])
    reserve = [base_min] * H
    charge_ok = [True] * H
    discharge_ok = [True] * H
    grid_cap: list[float | None] = [None] * H
    factor: list[float] = [1.0] * H

    for d in directives:
        t, adj = d["directive_type"], d["structured_adjustment"]
        if t == "no_op" or adj is None:
            continue
        for h in adj["hours"]:
            if t == "solar_reduction":
                factor[h] = min(factor[h], float(adj["factor"]))
            elif t == "minimum_battery_reserve":
                reserve[h] = max(reserve[h], float(adj["minimum_energy_kwh"]))
            elif t == "no_charge_window":
                charge_ok[h] = False
            elif t == "no_discharge_window":
                discharge_ok[h] = False
            elif t == "max_grid_window":
                cap = float(adj["max_grid_kwh"])
                grid_cap[h] = cap if grid_cap[h] is None else min(grid_cap[h], cap)

    eff = [solar[h] * factor[h] for h in range(H)]
    return Constraints(eff, reserve, charge_ok, discharge_ok, grid_cap)


def _solve_lp(hours: list[dict], battery: dict, cons: Constraints):
    n = 7 * H
    G, S, C, D, E, RS, GS = (k * H for k in range(7))
    demand = [float(h["demand_kwh"]) for h in hours]
    tariff = [float(h["tariff_bdt_per_kwh"]) for h in hours]
    cap = float(battery["capacity_kwh"])
    e0 = float(battery["initial_energy_kwh"])
    base_min = float(battery["minimum_energy_kwh"])
    max_c = float(battery["max_charge_kwh_per_hour"])
    max_d = float(battery["max_discharge_kwh_per_hour"])

    cost = np.zeros(n)
    for h in range(H):
        cost[G + h] = tariff[h]
        cost[C + h] = CYCLE_EPS
        cost[D + h] = CYCLE_EPS
        cost[RS + h] = BIG_M
        cost[GS + h] = BIG_M

    bounds = []
    bounds += [(0, None)] * H  # g
    bounds += [(0, max(0.0, cons.effective_solar[h])) for h in range(H)]  # s
    bounds += [(0, max_c if cons.charge_allowed[h] else 0) for h in range(H)]  # c
    bounds += [(0, max_d if cons.discharge_allowed[h] else 0) for h in range(H)]  # d
    bounds += [(min(base_min, cap), cap)] * H  # e
    bounds += [(0, None if cons.reserve[h] > base_min else 0) for h in range(H)]  # rs
    bounds += [(0, None if cons.grid_cap[h] is not None else 0) for h in range(H)]  # gs

    a_eq = lil_matrix((2 * H + 1, n))
    b_eq = np.zeros(2 * H + 1)
    for h in range(H):
        a_eq[h, G + h] = 1
        a_eq[h, S + h] = 1
        a_eq[h, D + h] = 1
        a_eq[h, C + h] = -1
        b_eq[h] = demand[h]
        r = H + h
        a_eq[r, E + h] = 1
        a_eq[r, C + h] = -1
        a_eq[r, D + h] = 1
        if h > 0:
            a_eq[r, E + h - 1] = -1
        else:
            b_eq[r] = e0
    a_eq[2 * H, E + H - 1] = 1
    b_eq[2 * H] = e0

    rows, b_ub = [], []
    for h in range(H):
        if cons.reserve[h] > base_min:
            rows.append({E + h: -1, RS + h: -1})
            b_ub.append(-cons.reserve[h])
        if cons.grid_cap[h] is not None:
            rows.append({G + h: 1, GS + h: -1})
            b_ub.append(cons.grid_cap[h])
    a_ub = None
    if rows:
        a_ub = lil_matrix((len(rows), n))
        for i, row in enumerate(rows):
            for j, v in row.items():
                a_ub[i, j] = v

    res = linprog(
        cost,
        A_ub=a_ub.tocsr() if a_ub is not None else None,
        b_ub=np.array(b_ub) if rows else None,
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if res.status != 0:
        return None
    x = res.x
    return {
        "s": x[S : S + H],
        "c": x[C : C + H],
        "d": x[D : D + H],
        "slack": float(x[RS : RS + H].sum() + x[GS : GS + H].sum()),
    }


def _r(v: float) -> float:
    v = round(float(v), ROUND)
    return 0.0 if v == 0 else v  # normalise -0.0


def _assemble(hours: list[dict], battery: dict, cons: Constraints, s, c, d) -> list[dict]:
    """Turn raw LP values into a replay-exact hourly plan (net battery flow, derived grid)."""
    e = float(battery["initial_energy_kwh"])
    plan = []
    for h in range(H):
        demand = float(hours[h]["demand_kwh"])
        net = _r(float(c[h]) - float(d[h]))  # >0 charge, <0 discharge
        solar = _r(min(max(float(s[h]), 0.0), cons.effective_solar[h]))
        grid = _r(demand + net - solar)
        if grid < 0:  # rounding / surplus solar: curtail instead of exporting
            solar = _r(solar + grid)
            grid = 0.0
        e = _r(e + net)
        action = "charge" if net > 0 else "discharge" if net < 0 else "idle"
        plan.append(
            {
                "hour": h,
                "grid_kwh": grid,
                "solar_used_kwh": solar,
                "battery_action": action,
                "battery_kwh": _r(abs(net)),
                "battery_energy_after_kwh": e,
            }
        )
    return plan


def optimize(hours: list[dict], battery: dict, directives: list[dict]) -> dict:
    cons = build_constraints(hours, battery, directives)
    sol = _solve_lp(hours, battery, cons)
    feasible = sol is not None
    if feasible:
        plan = _assemble(hours, battery, cons, sol["s"], sol["c"], sol["d"])
        slack_used = sol["slack"] > 1e-6
    else:  # safe fallback: battery idle all day, always energy-balanced and neutral
        zeros = [0.0] * H
        s = [min(cons.effective_solar[h], float(hours[h]["demand_kwh"])) for h in range(H)]
        plan = _assemble(hours, battery, cons, s, zeros, zeros)
        slack_used = False

    tariff = [float(h["tariff_bdt_per_kwh"]) for h in hours]
    return {
        "hourly_plan": plan,
        "total_grid_kwh": _r(sum(p["grid_kwh"] for p in plan)),
        "total_cost_bdt": _r(sum(p["grid_kwh"] * tariff[p["hour"]] for p in plan)),
        "peak_grid_kwh": _r(max(p["grid_kwh"] for p in plan)),
        "feasible": feasible,
        "slack_used": slack_used,
        "constraints": cons,
    }
