"""LLM client: interprets operator notes into raw directive JSON (OpenAI-compatible Chat Completions).

Works with Groq (default), OpenAI, Gemini's OpenAI endpoint, OpenRouter, or a local
OpenAI-compatible server (Ollama / vLLM). Configure with LLM_API_KEY, LLM_BASE_URL, LLM_MODEL.

LLM_API_KEY may hold several comma-separated keys. Every (key, model) pair has its own provider
rate-limit budget, so requests rotate across pairs and skip a pair while it is cooling down after
a 429.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

import httpx

SYSTEM_PROMPT = """You convert campus-energy operator notes into structured directives for a 24-hour battery/solar/grid scheduler (hours 0-23 of ONE day).

Each note maps to EXACTLY ONE of these directive types:
- solar_reduction: usable rooftop solar / PV output is reduced during some hours (cleaning, washing, shading, clouds, inverter work, maintenance, outage of panels). Give "factor" = fraction of forecast solar that REMAINS usable (0..1).
    "drop to about 20%" -> 0.2 ; "80% reduction" / "cut by 80%" -> 0.2 ; "one-fifth of normal" -> 0.2 ; "half" -> 0.5 ; "reduced by a quarter" -> 0.75 ; "no solar"/"panels offline" -> 0.
- minimum_battery_reserve: battery must keep at least some stored energy during some hours. Give "minimum_energy_kwh" if stated in energy units (convert MWh to kWh), OR "minimum_percent_of_capacity" if stated as a percent/fraction of battery capacity / state of charge (e.g. "half full" -> 50).
- no_charge_window: battery may NOT charge during some hours (charger isolated/offline/disabled, charging circuit unavailable, do not charge).
- no_discharge_window: battery may NOT discharge / supply energy during some hours (relay/protection testing, do not discharge, hold battery output).
- max_grid_window: grid import/intake/purchase must not exceed a stated kWh per hour during some hours (feeder / transformer / substation limit). Give "max_grid_kwh" (convert MW or MWh per hour to kWh: 0.2 MW -> 200).
- no_op: the note does not change today's energy schedule: unrelated campus news (menus, deadlines, bookings, notices, events), things scheduled for another day ("next week", "next month"), purely informational remarks, or anything that fits none of the types above. NEVER invent a constraint for an irrelevant note.

TIME RULES (critical):
- Output time as windows [start_hour, end_hour] on a 24-hour clock, start INCLUSIVE, end EXCLUSIVE.
  "1 PM to 3 PM" -> [13, 15] (hours 13 and 14). "from 6 PM until 9 PM" -> [18, 21]. "between 13:00 and 15:00" -> [13, 15].
  "noon" = 12, "midnight" = 0 (as an end time, midnight = 24). "from 10 PM until 2 AM" -> [22, 2].
  "from 11 AM for three hours" -> [11, 14]. "during the 3 PM hour" / "at 3 PM for one hour" -> [15, 16].
  "all day" / "throughout the day" -> [0, 24].
- When AM/PM is omitted, infer from context: solar work, cleaning, office activity -> daytime (e.g. "from one until three" -> [13, 15]); "evening" -> PM; "early morning"/"overnight" -> AM.
- Use several windows only if the note lists separate time ranges.

Do not change demand, tariff, or battery parameters. Do not output any other directive types.

Return ONLY a JSON object of this form (one entry per note, same order, note_index starting at 0):
{"interpretations": [
  {"note_index": 0, "directive_type": "solar_reduction", "windows": [[13, 15]], "factor": 0.2, "explanation": "short reason"},
  {"note_index": 1, "directive_type": "minimum_battery_reserve", "windows": [[18, 21]], "minimum_energy_kwh": 120, "explanation": "..."},
  {"note_index": 2, "directive_type": "minimum_battery_reserve", "windows": [[18, 21]], "minimum_percent_of_capacity": 50, "explanation": "..."},
  {"note_index": 3, "directive_type": "no_charge_window", "windows": [[14, 16]], "explanation": "..."},
  {"note_index": 4, "directive_type": "no_discharge_window", "windows": [[18, 20]], "explanation": "..."},
  {"note_index": 5, "directive_type": "max_grid_window", "windows": [[19, 21]], "max_grid_kwh": 180, "explanation": "..."},
  {"note_index": 6, "directive_type": "no_op", "explanation": "Unrelated to the energy schedule."}
]}"""


class LLMError(RuntimeError):
    pass


def _config() -> dict:
    primary = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    backups = os.getenv("LLM_FALLBACK_MODELS", "openai/gpt-oss-20b,qwen/qwen3.8-27b")
    models = [primary] + [m.strip() for m in backups.split(",") if m.strip() and m.strip() != primary]
    return {
        "api_keys": [k.strip() for k in (os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or "").split(",") if k.strip()],
        "base_url": os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
        "models": models,
        "timeout": float(os.getenv("LLM_TIMEOUT_SECONDS", "10")),
        "budget": float(os.getenv("LLM_TOTAL_BUDGET_SECONDS", "20")),
        "reasoning_effort": os.getenv("LLM_REASONING_EFFORT", "low"),
    }


def llm_configured() -> bool:
    return bool(_config()["api_keys"])


def model_name() -> str:
    return ",".join(_config()["models"])


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise LLMError("no JSON object in model output")
    return json.loads(text[start : end + 1])


def _reasoning_param(model: str, effort: str) -> dict:
    if not effort:
        return {}
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": effort}
    if model.startswith("qwen/"):
        return {"reasoning_effort": "none"}  # answer directly; keeps latency and token use low
    return {}


def _retry_after(r: httpx.Response) -> float:
    try:
        return float(r.headers.get("retry-after", "2"))
    except ValueError:
        return 2.0


def _call_model(cfg: dict, api_key: str, model: str, messages: list[dict], timeout: float) -> list[dict]:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": 1500,
        **_reasoning_param(model, cfg["reasoning_effort"]),
    }
    r = httpx.post(
        f"{cfg['base_url']}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=timeout,
    )
    if r.status_code == 429:
        raise RateLimited(_retry_after(r))
    if r.status_code != 200:
        raise LLMError(f"provider HTTP {r.status_code}")
    try:
        content = r.json()["choices"][0]["message"]["content"]
        data = _extract_json(content)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError(f"malformed model output ({type(exc).__name__})") from None
    items = data.get("interpretations") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise LLMError("model output missing 'interpretations' list")
    return items


class RateLimited(LLMError):
    def __init__(self, retry_after: float):
        super().__init__(f"rate limited (retry after {retry_after:.1f}s)")
        self.retry_after = retry_after


# (key position, model) -> monotonic time until which the pair is known to be rate limited
_COOLDOWN: dict[tuple[int, str], float] = {}
_COOLDOWN_LOCK = threading.Lock()


def _cooling(pair: tuple[int, str]) -> float:
    with _COOLDOWN_LOCK:
        return max(0.0, _COOLDOWN.get(pair, 0.0) - time.monotonic())


def interpret_notes_llm(notes: list[str], capacity: float, feedback: str | None = None) -> list[dict]:
    """Interpret all notes in one LLM call. Rotates through the configured (key, model) pairs on
    rate limits / provider errors (each pair has its own token budget) within a total time budget.
    Returns raw (unvalidated) interpretation dicts."""
    cfg = _config()
    if not cfg["api_keys"]:
        raise LLMError("LLM_API_KEY not configured")
    user = {
        "battery_capacity_kwh": capacity,
        "operator_notes": [{"note_index": i, "text": n} for i, n in enumerate(notes)],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user)},
    ]
    if feedback:
        messages.append(
            {"role": "user", "content": f"Your previous answer failed validation: {feedback}. Return corrected JSON."}
        )

    # model-major order: the strongest model is tried on every key before falling back
    pairs = [(k, m) for m in cfg["models"] for k in range(len(cfg["api_keys"]))]
    deadline = time.monotonic() + cfg["budget"]
    last: LLMError = LLMError("no model attempted")
    for _round in range(3):
        limited = False
        for pair in pairs:
            if _cooling(pair) > 0:
                limited = True
                last = RateLimited(_cooling(pair))
                continue
            remaining = deadline - time.monotonic()
            if remaining < 1.5:
                raise last
            key_pos, model = pair
            try:
                return _call_model(cfg, cfg["api_keys"][key_pos], model, messages, min(cfg["timeout"], remaining))
            except RateLimited as exc:
                with _COOLDOWN_LOCK:
                    _COOLDOWN[pair] = time.monotonic() + exc.retry_after
                limited = True
                last = exc
            except httpx.HTTPError as exc:
                last = LLMError(f"provider unreachable ({type(exc).__name__})")
            except LLMError as exc:
                last = exc
        if not limited:
            break
        pause = min(_cooling(p) for p in pairs)
        if time.monotonic() + pause + 2 > deadline:
            break
        time.sleep(max(pause, 0.2))
    raise last
