"""Paraphrase robustness check: 28 hidden-style notes (all directive types + distractors).

Usage: python scripts/paraphrase_check.py <base_url> [concurrency] [prefix to bypass cache]
"""
import json, pathlib, sys, time, concurrent.futures as cf
import httpx

BASE = sys.argv[1]
base = json.load(open(str(pathlib.Path(__file__).resolve().parents[1] / "samples" / "public_sample_cases.json")))["cases"][0]["input"]
CAP = base["battery"]["capacity_kwh"]

# (note, type, hours, value_key, value)
CASES = [
    ("PV production will drop to about 20% between 13:00 and 15:00.", "solar_reduction", [13, 14], "factor", 0.2),
    ("Panel washing from one until three will leave roughly one-fifth of normal solar output.", "solar_reduction", [13, 14], "factor", 0.2),
    ("Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window.", "solar_reduction", [13, 14], "factor", 0.2),
    ("Haze is expected to cut solar generation by 40 percent from 10 AM to noon.", "solar_reduction", [10, 11], "factor", 0.6),
    ("Only a quarter of the usual PV yield will be available between 11:00 and 14:00.", "solar_reduction", [11, 12, 13], "factor", 0.25),
    ("Inverter firmware update will take all solar offline from 9 to 11 in the morning.", "solar_reduction", [9, 10], "factor", 0.0),
    ("Solar arrays will run at 70% capacity from 12 PM until 4 PM due to dust.", "solar_reduction", [12, 13, 14, 15], "factor", 0.7),
    ("Keep at least 120 kWh in reserve from 6 PM until 9 PM.", "minimum_battery_reserve", [18, 19, 20], "minimum_energy_kwh", 120),
    ("The battery must hold no less than 0.15 MWh between 19:00 and 22:00 for emergency lighting.", "minimum_battery_reserve", [19, 20, 21], "minimum_energy_kwh", 150),
    ("Maintain the battery at least half full from 5 PM to 8 PM.", "minimum_battery_reserve", [17, 18, 19], "minimum_energy_kwh", CAP * 0.5),
    ("Security wants a 200 kWh backup kept in storage during 8-10 PM.", "minimum_battery_reserve", [20, 21], "minimum_energy_kwh", 200),
    ("Do not charge the battery between 2 PM and 4 PM.", "no_charge_window", [14, 15], None, None),
    ("The battery charger will be isolated for inspection from 10:00 to 13:00.", "no_charge_window", [10, 11, 12], None, None),
    ("Charging must be suspended during the 3 AM to 5 AM window.", "no_charge_window", [3, 4], None, None),
    ("Battery discharge is prohibited between 6 and 8 in the evening while protection relays are tested.", "no_discharge_window", [18, 19], None, None),
    ("Do not draw power from the battery from 7 AM to 9 AM.", "no_discharge_window", [7, 8], None, None),
    ("The battery cannot supply the campus from 16:00 to 18:00.", "no_discharge_window", [16, 17], None, None),
    ("Grid import must stay at or below 180 kWh per hour from 7 PM to 9 PM.", "max_grid_window", [19, 20], "max_grid_kwh", 180),
    ("The feeder is limited to 0.25 MW between 17:00 and 20:00.", "max_grid_window", [17, 18, 19], "max_grid_kwh", 250),
    ("Utility asks us to cap purchases from the grid at 150 kWh each hour from midnight to 3 AM.", "max_grid_window", [0, 1, 2], "max_grid_kwh", 150),
    ("The cafeteria menu changes tomorrow.", "no_op", None, None, None),
    ("Solar panel cleaning is scheduled for next week.", "no_op", None, None, None),
    ("The library will close early at 8 PM for exam preparation.", "no_op", None, None, None),
    ("Next month the battery will be upgraded to 800 kWh.", "no_op", None, None, None),
    ("A new tariff structure is being discussed by the finance committee.", "no_op", None, None, None),
    ("From 10 PM until 2 AM do not charge the battery.", "no_charge_window", [0, 1, 22, 23], None, None),
    ("Solar will be halved for the 11 o'clock hour.", "solar_reduction", [11], "factor", 0.5),
    ("Grid draw should not exceed 200 kWh from 18:00 through 20:00.", "max_grid_window", [18, 19], "max_grid_kwh", 200),
]


def run(i_note):
    i, (note, typ, hours, key, val) = i_note
    body = dict(base, scenario_id=f"STRESS-{i}-{time.time()}", operator_notes=[(sys.argv[3] + " " if len(sys.argv)>3 else "") + note])
    t = time.time()
    r = httpx.post(f"{BASE}/optimize-energy", json=body, timeout=40)
    dt = time.time() - t
    d = r.json()
    e = d["directive_interpretation"][0]
    got_t, adj = e["directive_type"], e["structured_adjustment"]
    ok = got_t == typ
    if ok and hours is not None:
        ok = adj["hours"] == hours
    if ok and key:
        ok = abs(adj[key] - val) <= 0.01
    return ok, dt, note, typ, hours, val, got_t, adj, r.status_code


with cf.ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as ex:
    res = list(ex.map(run, enumerate(CASES)))
fails = 0
for ok, dt, note, typ, hours, val, got_t, adj, sc in res:
    if not ok:
        fails += 1
        print(f"FAIL [{sc}] {dt:.1f}s {note!r}\n   want {typ} {hours} {val}\n   got  {got_t} {adj}")
lat = sorted(r[1] for r in res)
print(f"{len(res)-fails}/{len(res)} correct; p95 {lat[int(len(lat)*0.95)-1]:.2f}s max {lat[-1]:.2f}s")
