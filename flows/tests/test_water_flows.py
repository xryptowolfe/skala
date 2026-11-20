"""Smoke test: the full water opt -> freq -> thermo chain gives physically sane numbers.

Uses the small def2-SVP basis so it runs in about 3 minutes on a laptop CPU.
The first run downloads the skala-1.1 checkpoint from Hugging Face.
"""

import json

import pytest
from flows.water.__main__ import main

# The shared opt -> freq -> thermo fixture takes ~3 min on a laptop and longer on
# CI runners; lift the repository-wide 300 s per-test limit for this module.
pytestmark = pytest.mark.timeout(1800)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    run = main(["opt", "freq", "thermo", "--basis", "def2-svp", "--out", str(out)])
    return json.loads((run.dir / "results.json").read_text())


def test_optimization_converges_to_bent_water(results):
    opt = results["optimization"]
    assert opt["converged"]
    assert opt["r_OH1_A"] == pytest.approx(opt["r_OH2_A"], abs=1e-3)  # C2v kept
    assert 0.94 < opt["r_OH1_A"] < 0.99
    assert 100.0 < opt["angle_HOH_deg"] < 108.0


def test_three_real_modes_in_expected_windows(results):
    freq = results["frequencies"]
    assert freq["n_imaginary"] == 0
    bend, sym, asym = (m["freq_cm"] for m in freq["modes"])
    assert 1550 < bend < 1800
    assert 3650 < sym < asym < 4100
    assert 0.5 < freq["zpe_eV"] < 0.65


def test_thermo_entropy_close_to_experiment(results):
    assert results["thermo"]["S_J_molK"] == pytest.approx(188.8, abs=3.0)
    assert results["thermo"]["G_Ha"] < results["thermo"]["H_Ha"]


def test_settings_recorded(results):
    assert results["settings"]["basis"] == "def2-svp"
    assert results["wall_time_s"] > 0
