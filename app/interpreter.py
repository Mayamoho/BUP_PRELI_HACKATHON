"""LLM interpretation with strict mapping, bounded repair, and validated caching."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import threading
import time

from .guardrails import GuardrailError, validate_entry
from .llm import LLMError, interpret_notes_llm, llm_configured

_CACHE = OrderedDict()
_CACHE_MAX = 512
_CACHE_TTL = 3600
_LOCK = threading.Lock()


def _validate_all(raw_items: list, notes: list[str], capacity: float) -> tuple[list[dict], list[str]]:
    if not isinstance(raw_items, list) or len(raw_items) != len(notes):
        return [], ["Exactly one interpretation per note is required"]
    if any(not isinstance(item, dict) or type(item.get("note_index")) is not int
           or item["note_index"] != index for index, item in enumerate(raw_items)):
        return [], ["note_index must be complete, unique and in original order"]
    results, problems = [], []
    for i, item in enumerate(raw_items):
        try:
            results.append(validate_entry(item, i, capacity))
        except GuardrailError:
            problems.append(f"note_index {i}: invalid directive shape or value")
    return results, problems


def interpret(notes: list[str], capacity: float) -> tuple[list[dict], str]:
    if not llm_configured():
        raise LLMError("LLM_API_KEY not configured")
    key = (tuple(notes), float(capacity))
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and time.monotonic() - cached[0] < _CACHE_TTL:
            _CACHE.move_to_end(key)
            return deepcopy(cached[1]), "llm-cache"
    deadline = time.monotonic() + 23.0
    feedback = None
    for _ in range(2):
        remaining = deadline - time.monotonic()
        if remaining < 1.5:
            break
        raw = interpret_notes_llm(notes, capacity, feedback=feedback, budget_seconds=remaining)
        results, problems = _validate_all(raw, notes, capacity)
        if not problems:
            with _LOCK:
                _CACHE[key] = (time.monotonic(), deepcopy(results))
                _CACHE.move_to_end(key)
                while len(_CACHE) > _CACHE_MAX:
                    _CACHE.popitem(last=False)
            return results, "llm"
        feedback = "; ".join(problems)
    raise LLMError("Model interpretation failed guardrails")
