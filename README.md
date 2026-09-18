# GridWise LLM — BUP preliminary

A FastAPI service that interprets operator notes using a real generative model, validates each directive, minimizes 24-hour grid cost, and independently verifies the returned schedule.

Required endpoints: `GET /health` and `POST /optimize-energy`. The deployed judging API is `https://gridwise-bup-preli.vercel.app`, and the public fallback image is `docker.io/amininrohul/gridwise:1.1.0`. Source stays private during the event according to the organizer's timing rule.

## Run locally

Requires Python 3.12. From a clean machine:

```bash
git clone https://github.com/Mayamoho/BUP_PRELI_HACKATHON.git
cd BUP_PRELI_HACKATHON
git switch codex/verified-api-vercel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Put the real `LLM_API_KEY` in `.env`. Do not commit that file. The application loads `.env` automatically.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
.venv/bin/python scripts/run_samples.py http://127.0.0.1:8000
```

Require `10/10 passed`. The sample runner compares model interpretation to organizer ground truth, replays the schedule against that ground truth and checks optimal cost. Health checks local model configuration; it does not prove the provider key or quota works.

A full sample request can be extracted without editing JSON:

```bash
.venv/bin/python -c 'import json; print(json.dumps(json.load(open("samples/public_sample_cases.json"))["cases"][0]["input"]))' > /tmp/gridwise-sample.json
curl --fail-with-body http://127.0.0.1:8000/optimize-energy \
  -H 'Content-Type: application/json' --data-binary @/tmp/gridwise-sample.json
```

The first sample's optimal cost is 38,365 BDT. Complete reference requests and responses are in `samples/public_sample_cases.json`; equivalent optimal hourly actions are accepted.

## Model and configuration

| Variable | Default / purpose |
|---|---|
| `LLM_API_KEY` | Required secret; `GROQ_API_KEY` is also accepted |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | `qwen/qwen3.8-27b` — primary model |
| `LLM_FALLBACK_MODELS` | `openai/gpt-oss-120b,openai/gpt-oss-20b` — real model fallbacks; comma-separated |
| `LLM_REASONING_EFFORT` | `low` for GPT-OSS; Qwen uses `none` for instruct mode |
| `LLM_TIMEOUT_SECONDS` | 10 seconds per provider attempt |
| `LLM_TOTAL_BUDGET_SECONDS` | 20 seconds per extraction invocation, bounded by the shared 23-second interpretation deadline |
| `PORT` | 8000 for Docker; local Uvicorn uses its `--port` option |

The primary and fallback use the provider's Chat Completions JSON interface. The supplied key was verified against Groq; model availability depends on the account. Rate limits are real operational constraints: the observed account limit was 8,000 tokens per minute per tested model. Model rotation and bounded retries help, but sufficient quota is still needed for repeated hidden tests. The service does not purchase a plan or raise account limits automatically.

Every note must be interpreted by a real LLM before a successful plan can be returned. Provider failures and invalid model output produce controlled errors; there is no rule-only successful fallback. `app/fallback.py` remains a legacy helper for standalone regression tests and is not imported by the production interpretation path. Only validated model results are cached, for one hour, keyed by notes and battery capacity. Cache entries are deep-copied to prevent request contamination.

## Architecture

```mermaid
flowchart LR
    A[Scenario JSON] --> B[Strict Pydantic validation]
    B --> C[LLM extracts type, windows and values]
    C --> D[Deterministic mapping and numeric guardrails]
    D --> E[Hard hourly constraints]
    E --> F[Exact HiGHS linear program]
    F --> G[Independent schema and schedule replay]
    G --> H[Valid JSON response]
```

The LLM returns one entry per note in the original order. Guardrails expand half-open time windows, convert percentage reserves using battery capacity, and construct the exact response adjustment shape. Unsupported types, missing/duplicate note mappings, invalid ranges and invalid factors are rejected. Solar reduced by 80% leaves factor 0.2; 1 PM–3 PM means hours 13 and 14. The model never modifies base demand, tariff or battery parameters.

The optimizer uses 96 continuous variables: grid import `g`, used solar `s`, signed battery charge `q`, and battery energy `e` for each hour. It minimizes only grid cost:

```text
minimize sum(tariff[h] * g[h])
g[h] + s[h] - q[h] = demand[h]
e[h] = e[h-1] + q[h], with e[-1] = initial_energy
0 <= s[h] <= effective_solar[h]
0 <= g[h] <= active_grid_cap[h]
-active_discharge_limit[h] <= q[h] <= active_charge_limit[h]
active_reserve[h] <= e[h] <= capacity
e[23] = initial_energy
```

Positive `q` charges; negative `q` discharges. This prevents simultaneous actions without integer variables and is exact for the challenge's lossless battery model. There are no slack variables, penalty relaxations or idle-plan escape paths. Infeasible constraints return 422. Overlapping reserves use the maximum, grid caps the minimum, and solar factors the tightest fraction of original forecast; ask organizers if they clarify a different overlapping-solar policy.

Independent replay reconstructs directive effects without reusing solver matrices or bound arrays. It checks the response schema, every hour, battery transitions, end neutrality, energy balance, limits and totals. A failed replay returns 500 and never a successful-looking schedule.

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_samples.py http://127.0.0.1:8000
```

The automated suite covers all ten public optimum costs, 100 random integer instances checked against an independent exhaustive dynamic program, fractional values, invalid directives, infeasible constraints, corrupted schedules, invalid requests, note mappings and model failure behavior. Offline/API mock tests do not prove live language understanding. See [verification report](docs/VERIFICATION.md) for the checks actually run on this branch.

## Vercel deployment

Production API: **https://gridwise-bup-preli.vercel.app**

The service is deployed on Vercel with the native FastAPI entrypoint `app/main.py`. `.python-version` selects Python 3.12 and `vercel.json` sets a 30-second function duration. `.vercelignore` excludes secrets and development assets. The Groq credential is stored as a private Vercel environment variable and is not present in the source or image.

1. Import this GitHub repository into Vercel using an account with repository access. Select branch `codex/verified-api-vercel` for a review deployment, or the merged production branch later.
2. Use repository root and the FastAPI framework preset. Keep default build settings; do not configure a frontend output directory or a Uvicorn start command.
3. Set `LLM_API_KEY` as a secret, `LLM_BASE_URL=https://api.groq.com/openai/v1`, `LLM_MODEL=qwen/qwen3.8-27b`, `LLM_FALLBACK_MODELS=openai/gpt-oss-120b,openai/gpt-oss-20b`, and `LLM_REASONING_EFFORT=low`.
4. Deploy; ensure the submitted URL permits unauthenticated access to both judging endpoints. Redeploy when environment variables change.
5. From outside Vercel, check `/health` and run `scripts/run_samples.py https://gridwise-bup-preli.vercel.app`. Require 10/10 and measure latency with fresh notes, not only cache hits.

On 2026-09-18, the production URL returned healthy and passed all 10 public cases with live model interpretation, ground-truth replay and exact reference costs. Measured end-to-end latency was 0.61–1.34 seconds for that run. This evidence does not guarantee hidden-case behavior or future provider quota. Official guide: https://vercel.com/docs/frameworks/backend/fastapi

## Docker fallback

```bash
docker pull amininrohul/gridwise:1.1.0
docker run --rm -p 8000:8000 --env-file .env amininrohul/gridwise:1.1.0
```

The published public image runs as a non-root user, binds `0.0.0.0`, exposes port 8000 and has an HTTP health check. No credentials or sample answer pack are baked into it. Its immutable reference is:

`docker.io/amininrohul/gridwise@sha256:0eaa230dde11440d379c8fe712781ebafc044247a8f16112f6ecf008acea769a`

The tag was confirmed through Docker Hub's unauthenticated registry API. A local container returned healthy and produced the exact SAMPLE-01 optimum of 38,365 BDT through a real model call.

## Submission and limitations

Submit the public API base URL, event GitHub repository, this README/configuration, the exact pullable image reference above, and a video of at most three minutes. A narration script is in `docs/VIDEO_SCRIPT.md`. Keep the repository private during the event and follow organizer instructions for post-deadline publication.

Scoring: interpretation 25, constraints 25, optimization 10, API 10, reliability 10, deployment 10, documentation 10. The video is a tie-break, not base points. Local checks cannot guarantee hidden-test scores or qualification.

Malformed input returns 400, impossible interpreted constraints 422, missing model configuration makes health return 503, and model/internal/verification failure returns a controlled 500. There is a 256 KiB request limit, a bounded read deadline, and a 27-second processing deadline. Per-process caching does not survive serverless cold starts or share entries across instances. Adequate provider quota remains necessary.

Credits: team repository implementation, Codex-assisted review and hardening, BUP supplied challenge/sample pack; FastAPI/Starlette, Pydantic, HTTPX, NumPy/SciPy/HiGHS, Uvicorn, python-dotenv and pytest. Review and understand the logic before presenting it as the team's submission.
