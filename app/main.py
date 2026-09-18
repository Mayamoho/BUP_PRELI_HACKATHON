"""Strict GridWise API: invalid input is rejected, an invalid plan never becomes a 200 response, and a
language-model outage degrades to the guarded backup interpreter instead of failing the request."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import logging
import time

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .interpreter import interpret
from .llm import LLMError, llm_configured, model_name
from .models import Scenario
from .optimizer import InfeasiblePlan, optimize
from .replay import replay

load_dotenv()
log = logging.getLogger("gridwise")


@asynccontextmanager
async def lifespan(app):
    app.state.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gridwise")
    try:
        yield
    finally:
        app.state.executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="GridWise LLM", version="1.1.0", lifespan=lifespan)


def _error(status, message):
    return JSONResponse(status_code=status, content={"error": message})


def validate_request(body):
    return Scenario.model_validate(body).model_dump()


def _strict_json(raw):
    def bad_number(value):
        raise ValueError("Nonfinite JSON")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=bad_number, object_pairs_hook=unique_keys)


@app.exception_handler(Exception)
async def unhandled(request, exc):
    log.error("request_failed error_type=%s", type(exc).__name__)
    return _error(500, "internal error")


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    # Ready as soon as the process serves requests; without a key, notes use the guarded backup.
    return {"status": "ok"}


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"service": "GridWise LLM", "endpoints": ["GET /health", "POST /optimize-energy"],
            "llm_model": model_name(), "llm_configured": llm_configured()}


@app.post("/optimize-energy")
async def optimize_energy(request: Request):
    try:
        raw = bytearray()
        async with asyncio.timeout(2.0):
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > 262144:
                    return _error(400, "request too large")
        req = validate_request(_strict_json(raw))
    except (ValueError, ValidationError, UnicodeDecodeError, TimeoutError):
        return _error(400, "invalid request")
    try:
        async with asyncio.timeout(27.0):
            return await asyncio.get_running_loop().run_in_executor(request.app.state.executor, _process, req)
    except TimeoutError:
        return _error(500, "processing deadline exceeded")


def _process(req):
    started = time.perf_counter()
    try:
        directives, source = interpret(req["operator_notes"], req["battery"]["capacity_kwh"])
        result = optimize(req["hours"], req["battery"], directives)
    except LLMError:
        return _error(500, "language model unavailable or invalid interpretation")
    except InfeasiblePlan:
        return _error(422, "infeasible directives")
    response = {"scenario_id": req["scenario_id"], "directive_interpretation": directives,
                **{key: result[key] for key in ("hourly_plan", "total_grid_kwh", "total_cost_bdt", "peak_grid_kwh", "plan_summary")}}
    if replay(req, response):
        return _error(500, "schedule verification failed")
    log.info("optimization_complete source=%s elapsed_ms=%.1f", source, (time.perf_counter()-started)*1000)
    return JSONResponse(content=response)
