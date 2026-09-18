"""GridWise LLM - FastAPI service exposing GET /health and POST /optimize-energy."""
from __future__ import annotations

import json
import logging
import math
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .interpreter import interpret
from .llm import llm_configured, model_name
from .optimizer import H, optimize
from .replay import replay

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gridwise")

app = FastAPI(title="GridWise LLM", version="1.0.0")

BATTERY_FIELDS = (
    "capacity_kwh",
    "initial_energy_kwh",
    "minimum_energy_kwh",
    "max_charge_kwh_per_hour",
    "max_discharge_kwh_per_hour",
)
HOUR_FIELDS = ("demand_kwh", "solar_kwh", "tariff_bdt_per_kwh")


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message


def _num(v, where: str, allow_negative: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise RequestError(400, f"{where} must be a number")
    f = float(v)
    if not math.isfinite(f):
        raise RequestError(400, f"{where} must be finite")
    if f < 0 and not allow_negative:
        raise RequestError(422, f"{where} must be non-negative")
    return f


def validate_request(body) -> dict:
    if not isinstance(body, dict):
        raise RequestError(400, "request body must be a JSON object")
    for k in ("scenario_id", "operator_notes", "hours", "battery"):
        if k not in body:
            raise RequestError(400, f"missing field: {k}")

    sid = body["scenario_id"]
    if not isinstance(sid, str):
        raise RequestError(400, "scenario_id must be a string")

    notes = body["operator_notes"]
    if not isinstance(notes, list) or not 1 <= len(notes) <= 3:
        raise RequestError(400, "operator_notes must be an array of 1-3 strings")
    if not all(isinstance(n, str) and n.strip() for n in notes):
        raise RequestError(400, "operator_notes entries must be non-empty strings")
    notes = [n.strip()[:8000] for n in notes]

    hours = body["hours"]
    if not isinstance(hours, list) or len(hours) != H:
        raise RequestError(400, "hours must be an array of exactly 24 entries")
    clean_hours: dict[int, dict] = {}
    for i, h in enumerate(hours):
        if not isinstance(h, dict):
            raise RequestError(400, f"hours[{i}] must be an object")
        hr = h.get("hour")
        if isinstance(hr, bool) or not isinstance(hr, (int, float)) or hr != int(hr) or not 0 <= hr <= 23:
            raise RequestError(400, f"hours[{i}].hour must be an integer 0-23")
        hr = int(hr)
        if hr in clean_hours:
            raise RequestError(400, f"duplicate hour {hr}")
        entry = {"hour": hr}
        for f in HOUR_FIELDS:
            if f not in h:
                raise RequestError(400, f"hours[{i}].{f} missing")
            entry[f] = _num(h[f], f"hours[{i}].{f}", allow_negative=(f == "tariff_bdt_per_kwh"))
        clean_hours[hr] = entry

    bat = body["battery"]
    if not isinstance(bat, dict):
        raise RequestError(400, "battery must be an object")
    battery = {}
    for f in BATTERY_FIELDS:
        if f not in bat:
            raise RequestError(400, f"battery.{f} missing")
        battery[f] = _num(bat[f], f"battery.{f}")
    if battery["minimum_energy_kwh"] > battery["capacity_kwh"]:
        raise RequestError(422, "battery.minimum_energy_kwh exceeds capacity_kwh")
    if not battery["minimum_energy_kwh"] - 1e-9 <= battery["initial_energy_kwh"] <= battery["capacity_kwh"] + 1e-9:
        raise RequestError(422, "battery.initial_energy_kwh must be between minimum_energy_kwh and capacity_kwh")

    return {
        "scenario_id": sid,
        "operator_notes": notes,
        "hours": [clean_hours[h] for h in range(H)],
        "battery": battery,
    }


def _summary(directives: list[dict], result: dict) -> str:
    applied = [d["directive_type"] for d in directives if d["applies"]]
    ignored = sum(1 for d in directives if not d["applies"])
    parts = ["Applied " + ", ".join(applied) + "." if applied else "No operator note changed the energy model."]
    if ignored:
        parts.append(f"Ignored {ignored} unrelated note(s) as no_op.")
    moves = sum(1 for p in result["hourly_plan"] if p["battery_action"] != "idle")
    parts.append(
        f"LP optimizer uses available solar and shifts battery energy toward high-tariff hours "
        f"({moves} active battery hours), restoring the initial battery level by hour 23. "
        f"Grid cost {result['total_cost_bdt']:.2f} BDT."
    )
    if not result["feasible"]:
        parts.append("Directives were infeasible together; returned a safe idle-battery plan.")
    elif result["slack_used"]:
        parts.append("Some directive limits could not be fully met with the given data; violations were minimised.")
    return " ".join(parts)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):  # never leak stack traces or secrets
    log.error("unhandled error: %s", type(exc).__name__)
    return _error(500, "internal error")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "service": "GridWise LLM",
        "endpoints": ["GET /health", "POST /optimize-energy"],
        "llm_model": model_name(),
        "llm_configured": llm_configured(),
    }


@app.post("/optimize-energy")
async def optimize_energy(request: Request):
    raw = await request.body()
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return _error(400, "malformed JSON")
    try:
        req = validate_request(body)
    except RequestError as exc:
        return _error(exc.status, exc.message)
    return await run_in_threadpool(_process, req)


def _process(req: dict) -> JSONResponse:
    t0 = time.perf_counter()
    directives, source = interpret(req["operator_notes"], req["battery"]["capacity_kwh"])
    t1 = time.perf_counter()
    result = optimize(req["hours"], req["battery"], directives)
    response = {
        "scenario_id": req["scenario_id"],
        "directive_interpretation": directives,
        "hourly_plan": result["hourly_plan"],
        "total_grid_kwh": result["total_grid_kwh"],
        "total_cost_bdt": result["total_cost_bdt"],
        "peak_grid_kwh": result["peak_grid_kwh"],
        "plan_summary": _summary(directives, result),
    }
    violations = replay(req, response)  # final replay guardrail
    if violations:
        log.warning("replay found %d violation(s) for %s: %s", len(violations), req["scenario_id"], violations[:3])
    log.info(
        "scenario=%s source=%s llm_ms=%.0f opt_ms=%.0f cost=%.2f",
        req["scenario_id"], source, (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000, result["total_cost_bdt"],
    )
    return JSONResponse(content=response)
