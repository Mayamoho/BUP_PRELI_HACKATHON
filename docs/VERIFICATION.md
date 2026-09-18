# Deployment verification

Repository: Mayamoho/BUP_PRELI_HACKATHON. Branch: codex/verified-api-vercel. Prepared 2026-09-18.

- 184 automated tests passed, including ten public optimum comparisons, 100 independent dynamic-programming comparisons, all ten API cases with supplied model interpretations, strict guardrail checks, invalid-input checks, and rejection of model/replay failures. Legacy standalone parser tests remain in the suite; the production path does not call that parser.
- A fresh local HTTP service using actual Groq model calls passed 10/10 public cases. Configured primary: Qwen 3.8 27B, with GPT-OSS 120B and GPT-OSS 20B as model fallbacks. Interpretations matched reference semantics; all schedules passed ground-truth replay and matched reference optimal cost.
- Observed local request latencies were 0.61–1.70 seconds; nearest-rank p95 for these ten measurements is 1.70 seconds. This is a small public-case sample, not a guarantee of hidden-case or hosted latency.
- Separate earlier tests of the participant-workspace implementation hit the account's provider token limit. The team branch keeps bounded real-model rotation. Adequate account quota remains necessary under repeated/concurrent judging load.

| Public case | Cost BDT | Local seconds |
|---|---:|---:|
| SAMPLE-01 | 38365 | 1.09 |
| SAMPLE-02 | 42885 | 0.61 |
| SAMPLE-03 | 35480 | 0.91 |
| SAMPLE-04 | 40495 | 0.68 |
| SAMPLE-05 | 33950 | 0.62 |
| SAMPLE-06 | 34090 | 0.91 |
| SAMPLE-07 | 38550 | 0.86 |
| SAMPLE-08 | 37665 | 1.21 |
| SAMPLE-09 | 34873 | 0.71 |
| SAMPLE-10 | 41620 | 1.70 |

## Production deployment

Production API: `https://gridwise-bup-preli.vercel.app`

Deployment ID: `dpl_4oWPmzKUNLRS2LExsbnpC4hrCq7w`

On 2026-09-18, `/health` returned 200 and a fresh run of `scripts/run_samples.py` against the production URL passed 10/10 public cases. The runner checks semantic interpretation against organizer ground truth, independently replays every schedule with those directives, and rejects costs above the supplied optima.

| Public case | Cost BDT | Hosted seconds |
|---|---:|---:|
| SAMPLE-01 | 38365 | 0.82 |
| SAMPLE-02 | 42885 | 0.85 |
| SAMPLE-03 | 35480 | 0.89 |
| SAMPLE-04 | 40495 | 0.86 |
| SAMPLE-05 | 33950 | 0.61 |
| SAMPLE-06 | 34090 | 0.81 |
| SAMPLE-07 | 38550 | 1.34 |
| SAMPLE-08 | 37665 | 0.71 |
| SAMPLE-09 | 34873 | 0.82 |
| SAMPLE-10 | 41620 | 1.11 |

The hosted run had a maximum and nearest-rank p95 of 1.34 seconds. These ten requests are evidence for the public cases at that time; they do not guarantee hidden-case behavior, future latency, provider availability or quota.

## Container publication

Public tag: `docker.io/amininrohul/gridwise:1.1.0`

Immutable reference: `docker.io/amininrohul/gridwise@sha256:0eaa230dde11440d379c8fe712781ebafc044247a8f16112f6ecf008acea769a`

Docker Hub's unauthenticated registry API reported the tag active with the same digest. The local copy of that image returned 200 from `/health`; a real-model SAMPLE-01 request passed independent replay at the exact 38,365 BDT optimum. No provider credential is baked into the image.

The final walkthrough must still be uploaded with the submission. No qualification score is claimed before organizer evaluation.
