"""Run every public sample case against a running service and check interpretation, validity and cost.

Usage: python scripts/run_samples.py [BASE_URL]   (default http://localhost:8000)
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.replay import replay  # noqa: E402

TOL = 0.01
BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
CASES = json.loads((pathlib.Path(__file__).resolve().parents[1] / "samples/public_sample_cases.json").read_text())["cases"]


def same_directive(got: dict, exp: dict) -> bool:
    if (got["applies"], got["directive_type"]) != (exp["applies"], exp["directive_type"]):
        return False
    a, b = got["structured_adjustment"], exp["structured_adjustment"]
    if a is None or b is None:
        return a is b
    if set(a) != set(b) or a["hours"] != b["hours"]:
        return False
    return all(abs(a[k] - b[k]) <= TOL for k in a if k != "hours")


def main() -> int:
    print("health:", httpx.get(f"{BASE}/health", timeout=10).json())
    passed = 0
    lat = []
    for case in CASES:
        req, exp = case["input"], case["expected_output"]
        t = time.perf_counter()
        r = httpx.post(f"{BASE}/optimize-energy", json=req, timeout=35)
        lat.append(time.perf_counter() - t)
        body = r.json()
        issues = []
        if r.status_code != 200:
            issues.append(f"HTTP {r.status_code}: {body}")
        else:
            if body["scenario_id"] != req["scenario_id"]:
                issues.append("scenario_id mismatch")
            got_d = body["directive_interpretation"]
            if [d["note_index"] for d in got_d] != list(range(len(req["operator_notes"]))):
                issues.append("note_index order")
            for g, e in zip(got_d, exp["directive_interpretation"]):
                if not same_directive(g, e):
                    issues.append(f"note {e['note_index']}: got {g['directive_type']} {g['structured_adjustment']} "
                                  f"expected {e['directive_type']} {e['structured_adjustment']}")
            issues += replay(req, body, exp["directive_interpretation"])  # judge replays with ground truth
            if body["total_cost_bdt"] > exp["total_cost_bdt"] + TOL:
                issues.append(f"cost {body['total_cost_bdt']} > optimal {exp['total_cost_bdt']}")
        status = "PASS" if not issues else "FAIL"
        passed += not issues
        print(f"{status} {case['id']} {lat[-1]:.2f}s " + ("; ".join(issues) if issues else f"cost={body['total_cost_bdt']}"))
    lat.sort()
    print(f"\n{passed}/{len(CASES)} passed; max latency {lat[-1]:.2f}s; p95 {lat[math.ceil(0.95 * len(lat)) - 1]:.2f}s")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
