# GridWise LLM: Smart Campus Energy Optimizer

BUP CSE Fest 2026 Hackathon, Online Preliminary.

This is an HTTP API. It reads 1-3 natural-language **operator notes** and turns them into structured directives using an **LLM**. Deterministic **guardrails** check every directive before use. A **linear-programming optimizer** then produces the lowest-cost valid 24-hour grid/solar/battery schedule.

| | |
|---|---|
| Live API | https://bup-preli-hackathon.onrender.com (Render) |
| Health | `GET /health` returns `{"status":"ok"}` |
| Main | `POST /optimize-energy` |
| LLM | Groq `openai/gpt-oss-120b` (open-weight model on Groq; OpenAI-compatible Chat Completions, JSON mode, temperature 0, reasoning effort low) |
| Optimizer | Linear program solved by **HiGHS** through `scipy.optimize.linprog` (exact optimum) |
| Port | `8000` (override with `PORT`) |
| Docker image | `docker.io/kawser81/gridwise-llm:1.0.0` |

---

## 1. Architecture

```
operator_notes ──► LLM (Groq, JSON mode) ──► raw JSON {directive_type, windows, numbers}
                                                  │
                                                  ▼
                        Deterministic guardrails (app/guardrails.py)
                        • directive_type ∈ 6 allowed values
                        • note_index mapping: exactly one entry per note, in order
                        • windows [start,end) expanded to unique ascending hours 0-23 (wraps midnight)
                        • factor ∈ [0,1] (a percentage is converted), reserve ∈ [0,capacity], grid cap ≥ 0, all finite
                        • "% of capacity" → kWh computed in code, not by the LLM
                        • no_op ⇒ applies=false, adjustment=null; otherwise applies=true + exact shape
                        • on failure: one corrective LLM retry with the validation errors fed back
                                                  │
                                                  ▼
                        Optimizer (app/optimizer.py): LP over 24 h, HiGHS solver
                                                  │
                                                  ▼
                        Final replay (app/replay.py): re-checks every rule and directive on the plan
                                                  │
                                                  ▼
                              JSON response (interpretation + hourly_plan + totals)
```

**The LLM's role.** The LLM is the primary interpreter of every operator note. It classifies each note as one of `solar_reduction`, `minimum_battery_reserve`, `no_charge_window`, `no_discharge_window`, `max_grid_window` or `no_op`. It also extracts the time window and the numeric value. The LLM returns time *windows* (`[13,15]`), and code expands them to hour lists (`[13,14]`). This keeps the start-inclusive/end-exclusive convention exact.

**Rate limits, rotation and circuit breaker.** Each request walks an ordered list of targets (provider, key, model): the primary provider's models first (`gpt-oss-120b`, `gpt-oss-20b`, `qwen/qwen3.8-27b` on Groq), then any backup providers configured with `LLM2_*` / `LLM3_*` (for example Cerebras or Gemini, both OpenAI-compatible). Note that Groq rate limits apply per account, so extra keys only help if they come from different accounts.
- A target that returns HTTP 429 is skipped until its `retry-after` has passed. If every target is rate-limited, the request waits for the first to free up.
- A target that is unreachable or returns 5xx is skipped for 30 s, so a provider outage costs one timeout, not one per request.
- The whole LLM stage (including the corrective retry) shares one 22 s budget per request, so a response always arrives inside the judge's 30 s limit.

On a 28-note paraphrase stress set (all five directive types plus distractors, MW/MWh units, "half full", "one-fifth", midnight wrap) the service scored 28/28, with p95 latency 1.45 s at 4 concurrent requests.

**Safe failure / backup path.** Two cases trigger the backup path:
- the LLM provider is unreachable, rate-limited or times out;
- the LLM output still fails the guardrails after one retry.

In either case only the affected notes go to a deterministic backup parser (`app/fallback.py`). Its output passes through the **same guardrails**. If the backup parser can't produce a valid directive, the note becomes `no_op`. The service never invents a constraint and never crashes. Clean LLM results are cached in memory (LRU), so repeated identical requests are instant.

**Optimizer model** (per hour *h*; all variables ≥ 0):

```
minimise   Σ tariff[h]·grid[h]  (+1e-6·(charge+discharge) tie-breaker against useless cycling)
s.t.       grid + solar_used + discharge = demand + charge          (energy balance)
           E[h] = E[h-1] + charge[h] − discharge[h],  E[-1] = initial (transition)
           max(base_min, reserve directive) ≤ E[h] ≤ capacity       (bounds + minimum_battery_reserve)
           charge ≤ max_charge (0 in no_charge_window)
           discharge ≤ max_discharge (0 in no_discharge_window)
           solar_used ≤ solar·factor                                 (solar_reduction)
           grid ≤ max_grid_kwh in listed hours                       (max_grid_window)
           E[23] = initial                                           (end-of-day neutrality)
```

Charge and discharge in the same hour are netted into one `battery_action`. Grid is re-derived from the rounded values, so energy balance holds exactly. Totals (`total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh`) are recomputed from `hourly_plan`. If directives conflict and make the plan infeasible (the organizers say valid cases never do this), reserve and grid-cap violations get a very large penalty instead of failing. If even that is infeasible, the service returns a safe idle-battery plan.

On all 10 public samples the optimizer reaches the **reference optimal cost exactly**.

---

## 2. Quickstart (local, from a clean machine)

Requirements: Python 3.11+ and git. Docker is optional.

```bash
git clone https://github.com/Mayamoho/BUP_PRELI_HACKATHON.git
cd BUP_PRELI_HACKATHON
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env                 # then put your Groq key in .env (LLM_API_KEY=...)
set -a; source .env; set +a          # load env vars into the shell

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Get a free Groq key at <https://console.groq.com/keys>.

### Environment variables

| Name | Required | Default | Meaning |
|---|---|---|---|
| `LLM_API_KEY` | yes (for LLM) | none | API key for the OpenAI-compatible LLM provider (Groq). Several keys may be given comma-separated (`key1,key2`) to multiply rate-limit budget. `GROQ_API_KEY` is also accepted. |
| `LLM_BASE_URL` | no | `https://api.groq.com/openai/v1` | Any OpenAI-compatible endpoint (OpenAI, OpenRouter, a local Ollama/vLLM server). |
| `LLM_MODEL` | no | `openai/gpt-oss-120b` | Model identifier. |
| `LLM_FALLBACK_MODELS` | no | `openai/gpt-oss-20b,qwen/qwen3.8-27b` | Comma-separated backup models tried on rate limit / provider error (each Groq model has its own token budget). |
| `LLM2_BASE_URL`, `LLM2_API_KEY`, `LLM2_MODELS` | no | none | Optional backup provider tried after the primary one (all three must be set; `LLM3_*` ... `LLM5_*` work the same way). Example Cerebras: `https://api.cerebras.ai/v1`, `gpt-oss-120b`. Example Gemini: `https://generativelanguage.googleapis.com/v1beta/openai`, `gemini-2.5-flash-lite`. |
| `LLM_TOTAL_BUDGET_SECONDS` | no | `20` | Time allowed for one round of LLM attempts; the whole LLM stage is also capped at 22 s per request (keeps requests < 30 s). |
| `LLM_REASONING_EFFORT` | no | `low` | Reasoning effort sent to gpt-oss models; set empty for models that do not accept it. |
| `LLM_TIMEOUT_SECONDS` | no | `10` | Per-LLM-call timeout. |
| `PORT` | no | `8000` | HTTP port. |
| `WEB_CONCURRENCY` | no | `2` | Uvicorn workers (Docker image). |
| `LOG_LEVEL` | no | `INFO` | Logging level. Logs never include keys or prompts. |

If `LLM_API_KEY` is missing, the service still starts and answers using the backup parser. The LLM is the intended path, so set the key for judging.

---

## 3. Test it

### Deployed instance

```bash
curl -s https://bup-preli-hackathon.onrender.com/health
# {"status":"ok"}
python scripts/run_samples.py https://bup-preli-hackathon.onrender.com
```

Free Render instances sleep when idle; the first request after a pause can take ~50 s.

### Health (local)

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}
```

### One public sample

```bash
python -c 'import json;print(json.dumps(json.load(open("samples/public_sample_cases.json"))["cases"][0]["input"]))' > sample01.json
curl -s -X POST http://localhost:8000/optimize-energy \
     -H 'Content-Type: application/json' --data @sample01.json | python -m json.tool | head -40
```

Expected response shape (abridged):

```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {"note_index": 0, "applies": true, "directive_type": "solar_reduction",
     "structured_adjustment": {"hours": [12, 13], "factor": 0.25}, "explanation": "..."},
    {"note_index": 1, "applies": false, "directive_type": "no_op",
     "structured_adjustment": null, "explanation": "..."}
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 90.0, "solar_used_kwh": 0.0, "battery_action": "idle",
     "battery_kwh": 0.0, "battery_energy_after_kwh": 110.0}
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 175.0,
  "plan_summary": "..."
}
```

(Equivalent optimal schedules may differ hour by hour. The cost equals the reference optimum, 38365.)

### All 10 public samples (end-to-end, through the LLM)

With the server running:

```bash
python scripts/run_samples.py http://localhost:8000
```

For each case, the script checks four things:
1. every directive (type, hours, values) against the reference;
2. a full replay of the plan against the **reference** directives: energy balance, effective solar, battery bounds/rates/transitions, windows, grid cap, neutrality and totals;
3. that the cost is ≤ the reference optimal cost;
4. latency.

Expected: `10/10 passed`.

### Offline unit tests (no API key needed)

```bash
pytest -q
```

These cover:
- optimizer optimality on all samples;
- guardrail rejection of bad LLM output;
- the backup parser, including the paraphrase examples from Problem Statement §11.4;
- malformed-request handling (400);
- the end-to-end API with the LLM unavailable.

---

## 4. Docker fallback

```bash
docker pull docker.io/kawser81/gridwise-llm:1.0.0
docker run --rm -p 8000:8000 -e LLM_API_KEY=<your_groq_key> docker.io/kawser81/gridwise-llm:1.0.0
curl -s http://localhost:8000/health        # {"status":"ok"}
```

- The container binds `0.0.0.0:8000` and runs as a non-root user.
- **No secrets are baked into the image**; the key is only passed at runtime.
- Build it yourself: `docker build -t gridwise-llm:1.0.0 .`

---

## 5. API contract summary

- `GET /health` returns `200 {"status":"ok"}`.
- `POST /optimize-energy` returns `200` with `scenario_id`, `directive_interpretation`, `hourly_plan` (24 entries), `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh` and `plan_summary`.
- `400 {"error": "..."}` means malformed JSON or a structurally invalid request, for example:
  - missing fields;
  - not exactly 24 hours, or duplicate hours;
  - 0 or more than 3 notes, or empty notes;
  - non-numeric values.
- `422 {"error": "..."}` means the request is well-formed but semantically invalid: negative energy values, or initial energy outside [minimum, capacity].
- `500 {"error":"internal error"}` is a controlled error. It never includes a stack trace or secrets.

Numbers are rounded to 4 decimals. That is far below the 0.01 judge tolerance.

---

## 6. Project layout

```
app/main.py         FastAPI app, request validation, error handling, response assembly
app/llm.py          LLM client + system prompt (Groq / any OpenAI-compatible endpoint)
app/guardrails.py   Deterministic validation & normalisation of LLM output
app/interpreter.py  LLM → guardrails → retry → backup parser pipeline, LRU cache
app/fallback.py     Deterministic backup parser (only when the LLM fails)
app/optimizer.py    LP model + HiGHS solve + plan assembly
app/replay.py       Independent judge-style replay validator
scripts/run_samples.py  End-to-end public sample checker
tests/              pytest suite
samples/            Public sample cases (organizer-provided)
```

---

## 7. Dependencies & credits

- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/): HTTP server
- [SciPy](https://scipy.org/) `linprog` with the [HiGHS](https://highs.dev/) solver, plus NumPy: optimization
- [httpx](https://www.python-httpx.org/): LLM HTTP client
- [Groq](https://groq.com/) hosting OpenAI **gpt-oss-120b** (open-weight): operator-note interpretation
- pytest: tests
- AI coding assistant (Claude Code) was used during development, as the rulebook permits.

## 8. Known limitations

- **Free-tier rate limits.** Groq's free tier allows about 8k tokens/min per model per account, and one request uses about 1.4k tokens. Backup providers (`LLM2_*`) add capacity, but a sustained burst can still exhaust every target. Those requests fall back to the deterministic parser, which covers common phrasings but is less robust than the LLM. Responses stay valid.
- **Ambiguous times.** Times without AM/PM are resolved from context: solar or maintenance work means daytime. Truly ambiguous notes may be misread.
- **One directive per note.** Each note maps to exactly one directive, as the Problem Statement specifies. A note that mentions two rules is reduced to the dominant one.
- **Slack penalty on conflicts.** If a request contains contradictory hard directives, the reserve and grid-cap limits are softened with a large penalty instead of failing. The organizers state that valid scoring cases are feasible.
- **Per-process cache.** The cache is in memory and per process, so it resets on restart.

## 9. Secret handling

- Keys are read only from environment variables.
- `.env` is git-ignored and docker-ignored; only `.env.example` (with no values) is committed.
- Logs contain scenario id, interpretation source, timings and cost. They never contain keys, prompts or stack traces.
- Error responses are generic.
