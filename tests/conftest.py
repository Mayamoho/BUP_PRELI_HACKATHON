import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "samples/public_sample_cases.json").read_text())["cases"]


@pytest.fixture
def case():
    return json.loads(json.dumps(CASES[0]))
