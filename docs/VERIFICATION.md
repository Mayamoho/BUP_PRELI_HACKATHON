# Review-branch verification

Repository: Mayamoho/BUP_PRELI_HACKATHON. Branch: codex/verified-api-vercel. Prepared 2026-09-18.

- 184 automated tests passed, including ten public optimum comparisons, 100 independent dynamic-programming comparisons, all ten API cases with supplied model interpretations, strict guardrail checks, invalid-input checks, and rejection of model/replay failures. Legacy standalone parser tests remain in the suite; the production path does not call that parser.
- A fresh local HTTP service using actual Groq model calls passed 10/10 public cases. Configured primary: Qwen 3.8 27B, with GPT-OSS 20B as a model fallback. Interpretations matched reference semantics; all schedules passed ground-truth replay and matched reference optimal cost.
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

The final Docker image `gridwise-team:1.1.0` built successfully. Its `/health` endpoint returned 200, and a real-model SAMPLE-01 request passed ground-truth replay at the exact 38,365 BDT optimum.

Remaining submission work: publish the exact image tag/digest to a registry, deploy and test the public Vercel URL, and finalize/upload the video. No hosted endpoint, registry pullability or qualification score is claimed here.
