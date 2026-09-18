"""Operator-note interpretation pipeline: LLM -> deterministic guardrails -> (retry) -> backup parser."""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

from .fallback import interpret_note_fallback
from .guardrails import GuardrailError, validate_entry
from .llm import LLMError, interpret_notes_llm

log = logging.getLogger("gridwise.interpreter")

_CACHE: "OrderedDict[tuple, tuple[list[dict], str]]" = OrderedDict()
_CACHE_MAX = 512
# total LLM time per request; the judge times out at 30 s and the optimizer needs well under 1 s
REQUEST_BUDGET_SECONDS = 22.0
_LOCK = threading.Lock()


def _validate_all(raw_items: list, notes: list[str], capacity: float) -> tuple[list[dict | None], list[str]]:
    """Map LLM items to notes by note_index (falling back to position) and validate each."""
    by_index: dict[int, dict] = {}
    for pos, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        idx = item.get("note_index", pos)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = pos
        if 0 <= idx < len(notes) and idx not in by_index:
            by_index[idx] = item
    results: list[dict | None] = []
    problems: list[str] = []
    for i in range(len(notes)):
        item = by_index.get(i)
        if item is None:
            results.append(None)
            problems.append(f"note_index {i}: missing")
            continue
        try:
            results.append(validate_entry(item, i, capacity))
        except GuardrailError as exc:
            results.append(None)
            problems.append(f"note_index {i}: {exc}")
    return results, problems


def interpret(notes: list[str], capacity: float) -> tuple[list[dict], str]:
    """Return (directive_interpretation entries in note order, source label)."""
    key = (tuple(notes), float(capacity))
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return [dict(e) for e in _CACHE[key][0]], _CACHE[key][1]

    results: list[dict | None] = [None] * len(notes)
    source = "llm"
    started = time.monotonic()
    try:
        raw = interpret_notes_llm(notes, capacity, budget=REQUEST_BUDGET_SECONDS)
        results, problems = _validate_all(raw, notes, capacity)
        remaining = REQUEST_BUDGET_SECONDS - (time.monotonic() - started)
        if problems and remaining > 3:  # one corrective retry with guardrail feedback, within the budget
            log.info("guardrail rejected LLM output: %s", "; ".join(problems))
            try:
                raw2 = interpret_notes_llm(notes, capacity, feedback="; ".join(problems), budget=remaining)
                retry, _ = _validate_all(raw2, notes, capacity)
                results = [r if r is not None else r2 for r, r2 in zip(results, retry)]
            except LLMError as exc:  # keep the valid first-pass entries
                log.warning("corrective retry failed: %s", exc)
    except LLMError as exc:
        log.warning("LLM unavailable, using backup parser: %s", exc)
        source = "fallback"

    final: list[dict] = []
    for i, entry in enumerate(results):
        if entry is None:
            if source == "llm":
                source = "llm+fallback"
            try:
                entry = validate_entry(interpret_note_fallback(notes[i]), i, capacity)
            except GuardrailError:
                entry = validate_entry(
                    {"directive_type": "no_op", "explanation": "Could not be validated; no constraint applied."},
                    i,
                    capacity,
                )
        final.append(entry)

    if source == "llm":  # only cache clean LLM results so provider outages are retried next time
        with _LOCK:
            _CACHE[key] = (final, source)
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
    return [dict(e) for e in final], source
