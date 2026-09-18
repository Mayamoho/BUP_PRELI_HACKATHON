# Solution video — target 2 minutes 40 seconds

Record the real application and code. Replace the final verification statement with the evidence actually obtained; do not describe mocked tests as real-model tests. Do not show `.env` or credentials. The guide requires a video, not this script alone.

Record this script against the reviewed team branch. A separate earlier draft video exists in the participant workspace; it predates this branch and must not be treated as its current verification evidence.

**0:00–0:20 — Problem**

“GridWise plans campus energy for the next 24 hours. We must meet hourly demand using solar, grid electricity, and a battery. Operator notes can temporarily reduce solar, require reserves, prohibit charging or discharging, or cap grid imports. Our objective is minimum grid cost while obeying every constraint.”

Show the sample input and its operator notes.

**0:20–0:50 — Architecture**

“The pipeline separates language from mathematics. A real generative language model reads every note and emits one structured directive. Irrelevant notes become no-op. Deterministic validation checks the note mapping, directive type, time window, units, numeric ranges, and exact adjustment shape before any constraint reaches the optimizer.”

Show the README architecture diagram, `app/prompt.py`, and `app/models.py`.

**0:50–1:10 — Interpretation detail**

“A reduction by 80 percent leaves a usable solar factor of 0.2. A window from 1 PM to 3 PM applies to hours 13 and 14. Percentage reserves are converted using the supplied battery capacity. The model cannot change demand, tariffs, or the underlying battery parameters. Invalid output is retried or rejected safely.”

Show a directive interpretation in the sample response.

**1:10–1:45 — Exact optimization**

“We solve a 96-variable linear program with HiGHS through SciPy. Each hour has grid import, used solar, signed battery flow, and battery energy. Signed flow permits one battery action per hour. Equality constraints enforce energy balance and battery transitions. Bounds enforce availability, reserves, rate limits, and grid caps. The final battery energy must equal its starting energy, so the starting charge is not a free energy source.”

Show `app/optimizer.py` and the equations in README.

**1:45–2:10 — Verification**

“Before returning JSON, independent replay checks every constraint and total. The reviewed branch passes 184 automated tests, including 100 random comparisons with an exhaustive dynamic program. It passed all ten public cases through the production Vercel API and real language models at exact reference cost, with measured requests between 0.61 and 1.34 seconds.”

Show the test run and `docs/VERIFICATION.md`.

**2:10–2:40 — Run and delivery**

“Dependencies and configuration are documented in the README. The service runs on Vercel and from our public Docker image, and exposes the required health and optimization endpoints. The Groq key stays private in Vercel. The live test command checks semantics, schedule validity, cost, and latency. Bounded retries and two backup models support repeated judging requests.”

Show `/health`, one actual POST, and the production 10/10 results. Finish with `https://gridwise-bup-preli.vercel.app`, the event repository, and `amininrohul/gridwise:1.1.0`.
