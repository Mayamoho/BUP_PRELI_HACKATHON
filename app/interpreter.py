"""LLM interpretation with strict mapping, bounded repair, validated caching and a guarded backup.

The LLM is the primary interpreter. Only if it is unavailable, or its output still fails the
guardrails after the repair attempt, are the affected notes sent to the deterministic backup
parser, whose output passes the same guardrails (or becomes no_op). A valid request therefore
never fails because of a provider outage."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import threading
import time

import logging

from .fallback import interpret_note_fallback
from .guardrails import GuardrailError, validate_entry
from .llm import LLMError, interpret_notes_llm, llm_configured

log = logging.getLogger("gridwise.interpreter")

_CACHE = OrderedDict()
_CACHE_MAX = 512
_CACHE_TTL = 3600
_LOCK = threading.Lock()


def _validate_all(raw_items: list, notes: list[str], capacity: float) -> tuple[dict[int, dict], list[str]]:
    """Return {note_index: validated entry} for the entries that pass, plus the problems found."""
    if not isinstance(raw_items, list) or len(raw_items) != len(notes):
        return {}, ["Exactly one interpretation per note is required"]
    if any(not isinstance(item, dict) or type(item.get("note_index")) is not int
           or item["note_index"] != index for index, item in enumerate(raw_items)):
        return {}, ["note_index must be complete, unique and in original order"]
    results, problems = {}, []
    for i, item in enumerate(raw_items):
        try:
            results[i] = validate_entry(item, i, capacity)
        except GuardrailError:
            problems.append(f"note_index {i}: invalid directive shape or value")
    return results, problems


def _backup(note: str, index: int, capacity: float) -> dict:
    try:
        return validate_entry(interpret_note_fallback(note), index, capacity)
    except GuardrailError:
        return validate_entry({"directive_type": "no_op", "explanation": "Could not be validated; no constraint applied."},
                              index, capacity)


def interpret(notes: list[str], capacity: float) -> tuple[list[dict], str]:
    key = (tuple(notes), float(capacity))
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and time.monotonic() - cached[0] < _CACHE_TTL:
            _CACHE.move_to_end(key)
            return deepcopy(cached[1]), "llm-cache"
    deadline = time.monotonic() + 23.0
    feedback = None
    best: dict[int, dict] = {}
    try:
        if not llm_configured():
            raise LLMError("LLM_API_KEY not configured")
        for _ in range(2):
            remaining = deadline - time.monotonic()
            if remaining < 1.5:
                break
            raw = interpret_notes_llm(notes, capacity, feedback=feedback, budget_seconds=remaining)
            results, problems = _validate_all(raw, notes, capacity)
            for i, entry in results.items():
                best.setdefault(i, entry)
            if not problems:
                final = [results[i] for i in range(len(notes))]
                with _LOCK:
                    _CACHE[key] = (time.monotonic(), deepcopy(final))
                    _CACHE.move_to_end(key)
                    while len(_CACHE) > _CACHE_MAX:
                        _CACHE.popitem(last=False)
                return final, "llm"
            feedback = "; ".join(problems)
        log.warning("model interpretation failed guardrails; using backup for affected notes")
    except LLMError as exc:
        log.warning("language model unavailable (%s); using backup parser", type(exc).__name__)
    final = [best[i] if i in best else _backup(note, i, capacity) for i, note in enumerate(notes)]
    return final, "llm+fallback" if best else "fallback"
