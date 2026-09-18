"""Deterministic validation / normalisation of LLM directive output (Problem Statement section 08).

The LLM returns time *windows* (start inclusive, end exclusive) and raw numbers; this module
expands windows into hour lists, converts percentages, checks ranges, and emits the exact
structured_adjustment shape. Anything that cannot be validated is rejected (never invented).
"""
from __future__ import annotations

import math
from typing import Any

DIRECTIVE_TYPES = (
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
)


class GuardrailError(ValueError):
    pass


def _num(v: Any, name: str) -> float:
    if isinstance(v, bool) or v is None:
        raise GuardrailError(f"{name} missing")
    if isinstance(v, str):
        v = v.strip().rstrip("%").replace(",", "")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise GuardrailError(f"{name} not numeric") from None
    if not math.isfinite(f):
        raise GuardrailError(f"{name} not finite")
    return f


def _hour(v: Any, allow_24: bool = False) -> int:
    f = _num(v, "hour")
    if abs(f - round(f)) > 1e-9:
        raise GuardrailError("hour not whole")
    h = int(round(f))
    if h == 24 and allow_24:
        return 24
    if h == 24:
        h = 0
    if not 0 <= h <= 23:
        raise GuardrailError("hour out of range")
    return h


def expand_window(start: Any, end: Any) -> list[int]:
    """[start, end) on a 24h clock; wraps past midnight (22 -> 2 gives 22,23,0,1)."""
    s = _hour(start)
    e = _hour(end, allow_24=True)
    if e == s:
        raise GuardrailError("empty window")
    if e > s:
        return list(range(s, e))
    return list(range(s, 24)) + list(range(0, e))


def normalize_hours(raw: dict) -> list[int]:
    hours: set[int] = set()
    windows = raw.get("windows")
    if isinstance(windows, list) and windows:
        for w in windows:
            if isinstance(w, dict):
                hours.update(expand_window(w.get("start_hour"), w.get("end_hour")))
            elif isinstance(w, (list, tuple)) and len(w) == 2:
                hours.update(expand_window(w[0], w[1]))
            else:
                raise GuardrailError("bad window")
    elif raw.get("start_hour") is not None and raw.get("end_hour") is not None:
        hours.update(expand_window(raw["start_hour"], raw["end_hour"]))
    elif isinstance(raw.get("hours"), list) and raw["hours"]:
        hours.update(_hour(h) for h in raw["hours"])
    else:
        raise GuardrailError("no hours")
    return sorted(hours)


def validate_entry(raw: dict, note_index: int, capacity: float) -> dict:
    """Validate one LLM interpretation; return the exact response-schema entry."""
    if not isinstance(raw, dict):
        raise GuardrailError("entry not an object")
    t = str(raw.get("directive_type", "")).strip().lower()
    if t not in DIRECTIVE_TYPES:
        raise GuardrailError(f"unsupported directive_type {t!r}")
    explanation = str(raw.get("explanation") or "").strip()[:300]

    if t == "no_op":
        return {
            "note_index": note_index,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": explanation or "This note does not affect today's 24-hour energy schedule.",
        }

    adj: dict[str, Any] = {"hours": normalize_hours(raw)}
    if t == "solar_reduction":
        if raw.get("factor") is not None:
            f = _num(raw["factor"], "factor")
            if 1 < f <= 100:  # percent given instead of fraction
                f /= 100
        elif raw.get("reduction_percent") is not None:
            f = 1 - _num(raw["reduction_percent"], "reduction_percent") / 100
        else:
            raise GuardrailError("factor missing")
        if not 0 <= f <= 1:
            raise GuardrailError("factor out of [0,1]")
        adj["factor"] = round(f, 6)
    elif t == "minimum_battery_reserve":
        if raw.get("minimum_energy_kwh") is not None:
            v = _num(raw["minimum_energy_kwh"], "minimum_energy_kwh")
        elif raw.get("minimum_percent_of_capacity") is not None:
            v = _num(raw["minimum_percent_of_capacity"], "minimum_percent_of_capacity") / 100 * capacity
        else:
            raise GuardrailError("minimum_energy_kwh missing")
        if v < 0 or v > capacity + 1e-9:
            raise GuardrailError("reserve outside [0, capacity]")
        adj["minimum_energy_kwh"] = round(v, 6)
    elif t == "max_grid_window":
        v = _num(raw.get("max_grid_kwh"), "max_grid_kwh")
        if v < 0:
            raise GuardrailError("negative grid cap")
        adj["max_grid_kwh"] = round(v, 6)

    return {
        "note_index": note_index,
        "applies": True,
        "directive_type": t,
        "structured_adjustment": adj,
        "explanation": explanation or f"Interpreted as {t}.",
    }
