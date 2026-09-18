"""Deterministic backup interpreter, used ONLY when the LLM provider is unavailable or its output
fails guardrails even after a retry. The LLM remains the primary interpretation path.

Produces the same raw shape as the LLM (directive_type + windows + numbers) so it goes through
the identical guardrail validation.
"""
from __future__ import annotations

import re

WORDNUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
FRACTIONS = {
    "three-quarters": 0.75, "three quarters": 0.75, "two-thirds": 2 / 3, "two thirds": 2 / 3,
    "two-fifths": 0.4, "one-fifth": 0.2, "one fifth": 0.2, "a fifth": 0.2,
    "one-third": 1 / 3, "one third": 1 / 3, "a third": 1 / 3,
    "one-quarter": 0.25, "one quarter": 0.25, "a quarter": 0.25,
    "one-tenth": 0.1, "a tenth": 0.1, "half": 0.5,
}

_T = r"(noon|midday|midnight|\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?|" + "|".join(WORDNUM) + r")"
RANGE = re.compile(
    rf"{_T}\s*(?:o'clock\s*)?(?:to|until|till|til|through|and|-|–|—)\s*{_T}",
    re.I,
)


def _parse_time(tok: str, is_end: bool) -> tuple[int, bool]:
    """Return (hour, explicit) where explicit means AM/PM, hh:mm, or 24h clock was given."""
    t = tok.strip().lower().replace(".", "")
    if t in ("noon", "midday"):
        return 12, True
    if t == "midnight":
        return (24 if is_end else 0), True
    if t in WORDNUM:
        return WORDNUM[t], False
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t)
    h = int(m.group(1))
    mer = m.group(3)
    if mer == "pm" and h != 12:
        h += 12
    elif mer == "am" and h == 12:
        h = 0
    explicit = bool(mer) or m.group(2) is not None or h > 12
    return h, explicit


def extract_windows(text: str) -> list[list[int]]:
    m = RANGE.search(text)
    if not m:
        return []
    s, s_exp = _parse_time(m.group(1), False)
    e, e_exp = _parse_time(m.group(2), True)
    if not e_exp:
        tail = re.match(r"\s*(a\.?m|p\.?m)", text[m.end():], re.I)
        if tail:
            e_exp = True
            if tail.group(1).lower().startswith("p") and e < 12:
                e += 12
            elif e == 12:
                e = 0
    if not e_exp and e < 7:
        e += 12  # bare "until three" -> afternoon
    if not s_exp:
        if s + 12 <= e and s < 12 and (e >= 13 or not e_exp):
            s += 12  # "1-3 PM" / "from one until three"
    return [[s % 24, 24 if e == 24 else e % 24]]


def _percent(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", text, re.I)
    return float(m.group(1)) if m else None


def _kwh(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(k|m)wh?\b", text, re.I)
    if not m:
        return None
    v = float(m.group(1))
    return v * 1000 if m.group(2).lower() == "m" else v


def interpret_note_fallback(note: str) -> dict:
    t = note.lower()
    windows = extract_windows(note)
    if not windows:
        return {"directive_type": "no_op", "explanation": "No schedulable time window found (backup parser)."}
    base = {"windows": windows, "explanation": "Interpreted by deterministic backup parser (LLM unavailable)."}

    if re.search(r"solar|\bpv\b|photovoltaic|panel", t):
        pct = _percent(t)
        factor = None
        if pct is not None:
            is_reduction = re.search(
                rf"{pct:g}\s*(?:%|percent)\s*(?:reduction|drop|cut|decrease|loss|lower)"
                r"|(?:reduc\w*|cut|decrease\w*|lower\w*|drop\w*)\s+(?:of|by)\s+(?:about\s+|roughly\s+|around\s+)?\d",
                t,
            )
            factor = 1 - pct / 100 if is_reduction else pct / 100
        else:
            for word, val in FRACTIONS.items():
                if word in t:
                    factor = val
                    break
            if factor is None and re.search(r"offline|unavailable|no solar|zero|shut ?down", t):
                factor = 0.0
        if factor is not None:
            return {**base, "directive_type": "solar_reduction", "factor": factor}

    if re.search(r"grid|import|intake|feeder|transformer|substation|utility", t) and _kwh(t) is not None:
        return {**base, "directive_type": "max_grid_window", "max_grid_kwh": _kwh(t)}

    if re.search(r"reserve|at least|keep|remain|minimum|maintain|no less", t) and re.search(r"batter|stor|reserve", t):
        k = _kwh(t)
        if k is not None:
            return {**base, "directive_type": "minimum_battery_reserve", "minimum_energy_kwh": k}
        pct = _percent(t)
        if pct is not None:
            return {**base, "directive_type": "minimum_battery_reserve", "minimum_percent_of_capacity": pct}
        if "half" in t:
            return {**base, "directive_type": "minimum_battery_reserve", "minimum_percent_of_capacity": 50}

    if re.search(r"discharg|battery output|draw\w* (?:energy )?from the battery", t):
        return {**base, "directive_type": "no_discharge_window"}
    if re.search(r"charg", t):
        return {**base, "directive_type": "no_charge_window"}

    return {"directive_type": "no_op", "explanation": "Note does not match a supported directive (backup parser)."}
