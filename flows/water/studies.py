"""Comparative studies built on the water flows.

Run from the repository root:
    python -m flows.water.studies functionals   # Skala versions vs LDA/GGA/meta-GGA
    python -m flows.water.studies energetics    # atomization, O-H bond, ionization (UKS)
    python -m flows.water.studies dimer         # (H2O)2 hydrogen bond, D3 and counterpoise
    python -m flows.water.studies basis         # basis-set convergence and density fitting
    python -m flows.water.studies solvation     # PCM solvation energy in several solvents
Add --device cuda to run on a GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data.s22 import create_s22_system, get_interaction_energy_s22
from ase.optimize import BFGS
from ase.units import Debye
from ase.vibrations import Vibrations

from . import flows
from . import reference as ref
from .core import (
    EV_TO_CM,
    HARTREE_EV,
    Run,
    Settings,
    geometry_summary,
    load_water,
    make_calculator,
)

KCAL_PER_HARTREE = 627.509474
KCAL_PER_EV = KCAL_PER_HARTREE / HARTREE_EV


class Study:
    """One output folder runs/<stamp>_study-<name>/ holding report.txt and results.json."""

    def __init__(self, name: str, settings: Settings, root: Path) -> None:
        self.name, self.settings = name, settings
        self.dir = root / f"{time.strftime('%Y%m%d-%H%M%S')}_study-{name}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.lines: list[str] = []
        self.results: dict = {}
        self.t0 = time.time()

    def say(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def sub_run(self, s: Settings, tag: str) -> Run:
        """A quiet flows Run inside this study folder (its own report.txt still written)."""
        return Run(settings=s, flows=[tag], root=self.dir)

    def finish(self) -> None:
        self.results["wall_time_s"] = round(time.time() - self.t0, 1)
        self.say(f"\nWall time {self.results['wall_time_s']} s. Results in {self.dir}/")
        (self.dir / "results.json").write_text(
            json.dumps(self.results, indent=2) + "\n"
        )
        (self.dir / "report.txt").write_text("\n".join(self.lines) + "\n")


def _quiet(fn, *args, **kwargs):
    """Run a flow without echoing its report; the sub-run's own report.txt keeps it."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _relax(atoms: Atoms, s: Settings, logfile: Path) -> int:
    opt = BFGS(atoms, logfile=str(logfile))
    if not opt.run(fmax=s.fmax, steps=200):
        raise RuntimeError(f"optimization did not converge ({logfile})")
    return opt.nsteps


# --------------------------------------------------------------------------- functionals


FUNCTIONALS = [
    "spw92",
    "pbe",
    "tpss",
    "scan",
    "r2scan",
    "skala-1.0",
    "skala-1.1-rev0",
    "skala-1.1",
]


def functionals(s: Settings, study: Study) -> None:
    """Optimize + harmonic frequencies of H2O with each functional (D3 off for all)."""
    s = replace(s, dispersion=False)
    study.say(
        f"== Functional comparison: opt + freq, {s.basis}, D3 off for every functional =="
    )
    study.say(
        f"{'functional':<16}{'O-H A':>8}{'HOH deg':>9}{'bend':>8}{'sym':>8}{'asym':>8}"
        f"{'MAE cm-1':>10}{'time s':>8}"
    )
    harm = np.array([m[1] for m in ref.MODES])
    rows = {}
    for xc in FUNCTIONALS:
        sx = replace(s, xc=xc)
        t = time.time()
        run = study.sub_run(sx, xc)
        atoms = load_water(sx)
        _quiet(flows.optimize, atoms, sx, run)
        _quiet(flows.frequencies, atoms, sx, run)
        _quiet(run.finish)
        g = run.results["optimization"]
        w = np.array([m["freq_cm"] for m in run.results["frequencies"]["modes"]])
        mae = float(np.mean(np.abs(w - harm)))
        rows[xc] = {
            **g,
            "freq_cm": w.tolist(),
            "mae_vs_harmonic_cm": mae,
            "time_s": time.time() - t,
        }
        study.say(
            f"{xc:<16}{g['r_OH1_A']:8.4f}{g['angle_HOH_deg']:9.2f}{w[0]:8.1f}{w[1]:8.1f}"
            f"{w[2]:8.1f}{mae:10.1f}{time.time() - t:8.0f}"
        )
    study.say(
        f"{'experiment':<16}{ref.R_OH:8.4f}{ref.ANGLE_HOH:9.2f}{harm[0]:8.1f}{harm[1]:8.1f}"
        f"{harm[2]:8.1f}{'':>10}"
    )
    study.say(
        "MAE: mean absolute error of the three frequencies vs experimental harmonic values."
    )
    study.results["functionals"] = rows


# --------------------------------------------------------------------------- energetics


def _species(
    symbols: str, positions, charge: int = 0, mult: int = 1, s: Settings | None = None
):
    atoms = Atoms(symbols, positions=positions)
    atoms.calc = make_calculator(s, charge=charge, multiplicity=mult)
    return atoms


def _diatomic_zpe(atoms: Atoms, s: Settings, name: Path) -> tuple[float, float]:
    """Harmonic frequency (cm-1) and ZPE (eV) of a diatomic from finite differences."""
    vib = Vibrations(atoms, name=str(name), delta=s.fd_step, nfree=2)
    vib.run()
    w = (
        float(np.max(np.abs(vib.get_energies()))) * EV_TO_CM
    )  # stretch is the only real mode
    return w, 0.5 * w / EV_TO_CM


def energetics(s: Settings, study: Study) -> None:
    """Atomization energy, first O-H bond energy and ionization energy (open shells use UKS)."""
    study.say(
        f"== Energetics with {s.xc}/{s.basis} (D3={s.dispersion}); open shells use UKS =="
    )

    # H2O: opt + freq for ZPE (reuse the flows)
    run = study.sub_run(s, "h2o")
    water = load_water(s)
    _quiet(flows.optimize, water, s, run)
    _quiet(flows.frequencies, water, s, run)
    _quiet(run.finish)
    e_h2o = water.get_potential_energy()
    zpe_h2o = run.results["frequencies"]["zpe_eV"]
    study.say(f"  H2O   E {e_h2o / HARTREE_EV:.8f} Ha   ZPE {zpe_h2o:.4f} eV")

    # Atoms
    e_o = _species("O", [[0, 0, 0]], mult=3, s=s).get_potential_energy()
    e_h = _species("H", [[0, 0, 0]], mult=2, s=s).get_potential_energy()
    study.say(
        f"  O(3P) E {e_o / HARTREE_EV:.8f} Ha   H(2S) E {e_h / HARTREE_EV:.8f} Ha"
    )

    # OH radical: opt + harmonic frequency
    oh = _species("OH", [[0, 0, 0], [0, 0, 0.97]], mult=2, s=s)
    _relax(oh, s, study.dir / "oh_opt.log")
    e_oh = oh.get_potential_energy()
    w_oh, zpe_oh = _diatomic_zpe(oh, s, study.dir / "oh_vib")
    study.say(
        f"  OH    E {e_oh / HARTREE_EV:.8f} Ha   r {oh.get_distance(0, 1):.4f} A   "
        f"w {w_oh:.1f} cm-1 (exp. {ref.OH_HARMONIC_CM:.1f})"
    )

    # H2O+ : vertical (neutral geometry) and adiabatic (relaxed cation, electronic only)
    cat = _species(
        water.get_chemical_symbols(), water.get_positions(), charge=1, mult=2, s=s
    )
    e_vert = cat.get_potential_energy()
    _relax(cat, s, study.dir / "h2o+_opt.log")
    e_adia = cat.get_potential_energy()
    gc = geometry_summary(cat)
    study.say(
        f"  H2O+  vertical E {e_vert / HARTREE_EV:.8f} Ha   relaxed E {e_adia / HARTREE_EV:.8f} Ha"
        f"   (O-H {gc['r_OH1_A']:.4f} A, HOH {gc['angle_HOH_deg']:.1f} deg)"
    )

    de = (e_o + 2 * e_h - e_h2o) * KCAL_PER_EV
    d0 = de - zpe_h2o * KCAL_PER_EV
    bde_e = (e_oh + e_h - e_h2o) * KCAL_PER_EV
    bde_0 = bde_e - (zpe_h2o - zpe_oh) * KCAL_PER_EV
    ie_v, ie_a = e_vert - e_h2o, e_adia - e_h2o
    table = [
        ("Atomization D0 (kcal/mol)", d0, ref.ATOMIZATION_D0_KCAL),
        ("  (De, no ZPE)", de, None),
        ("O-H bond D0(H-OH) (kcal/mol)", bde_0, ref.BDE_D0_KCAL),
        ("Ionization, adiabatic (eV)", ie_a, ref.IE_ADIABATIC_EV),
        ("Ionization, vertical (eV)", ie_v, ref.IE_VERTICAL_EV),
    ]
    study.say(f"\n  {'quantity':<30}{s.xc:>12}{'experiment':>12}{'error':>9}")
    for label, val, exp in table:
        tail = f"{exp:12.2f}{val - exp:9.2f}" if exp is not None else ""
        study.say(f"  {label:<30}{val:12.2f}{tail}")
    study.say(
        "  Adiabatic IE is electronic only (no ZPE change). Atomic spin-orbit (~0.2 kcal/mol) ignored."
    )
    study.results["energetics"] = {
        "E_Ha": {
            "H2O": e_h2o / HARTREE_EV,
            "O": e_o / HARTREE_EV,
            "H": e_h / HARTREE_EV,
            "OH": e_oh / HARTREE_EV,
            "H2O+_vertical": e_vert / HARTREE_EV,
            "H2O+_relaxed": e_adia / HARTREE_EV,
        },
        "zpe_eV": {"H2O": zpe_h2o, "OH": zpe_oh},
        "atomization_D0_kcal": d0,
        "atomization_De_kcal": de,
        "bde_D0_kcal": bde_0,
        "ie_adiabatic_eV": ie_a,
        "ie_vertical_eV": ie_v,
        "oh_harmonic_cm": w_oh,
    }


# --------------------------------------------------------------------------- dimer


def _pyscf_energy(
    symbols, positions, ghosts: set[int], xc: str, s: Settings, d3: bool
) -> float:
    """Total energy (Ha) with ghost atoms (basis functions, no nucleus/electrons) at `ghosts`."""
    from skala.pyscf import SkalaKS
    from skala.pyscf.retry import retry_scf

    from pyscf import gto

    atom = [
        (("ghost-" + sym) if i in ghosts else sym, tuple(p))
        for i, (sym, p) in enumerate(zip(symbols, positions, strict=True))
    ]
    mol = gto.M(atom=atom, basis=s.basis, verbose=0)
    ks = SkalaKS(
        mol, xc=xc, with_dftd3=d3, device=None if s.device == "cpu" else _cuda()
    )
    ks, _ = retry_scf(ks)  # same convergence fallbacks as the ASE calculator
    e = ks.e_tot
    if not ks.converged:
        raise RuntimeError(f"SCF did not converge for {xc}")
    return float(e)


def _cuda():
    import torch

    return torch.device("cuda")


def dimer(s: Settings, study: Study) -> None:
    """Interaction energy of the S22 water dimer: raw / counterpoise, D3 off / on, two bases.

    Counterpoise needs ghost atoms (basis functions without nucleus or electrons). Skala's
    SCF does not converge with ghost atoms, so for Skala the basis-set superposition error
    is judged instead from the raw value in the larger def2-QZVP basis.
    """
    d = create_s22_system("Water_dimer")
    sym, pos = d.get_chemical_symbols(), d.get_positions()
    a, b = set(range(3)), set(range(3, 6))
    sym_a, pos_a = [sym[i] for i in sorted(a)], pos[sorted(a)]
    sym_b, pos_b = [sym[i] for i in sorted(b)], pos[sorted(b)]
    ref_kcal = get_interaction_energy_s22("Water_dimer") * KCAL_PER_EV
    big = "def2-qzvp"
    study.say(
        f"== Water dimer (S22 geometry); reference CCSD(T)/CBS {ref_kcal:.2f} kcal/mol =="
    )
    study.say(
        f"  {'functional':<12}{'raw':>8}{'CP':>8}{'raw+D3':>8}{'CP+D3':>8}{'QZVP+D3':>9}"
        f"{'D3 part':>9}   kcal/mol, {s.basis} unless noted"
    )

    def e_int(xc: str, sb: Settings, d3: bool) -> float:
        e_ab = _pyscf_energy(sym, pos, set(), xc, sb, d3)
        e_a = _pyscf_energy(sym_a, pos_a, set(), xc, sb, d3)
        e_b = _pyscf_energy(sym_b, pos_b, set(), xc, sb, d3)
        return (e_ab - e_a - e_b) * KCAL_PER_HARTREE

    rows = {}
    for xc in ["pbe", "r2scan", s.xc]:
        raw = e_int(xc, s, False)
        raw_d3 = e_int(xc, s, True)
        d3_part = raw_d3 - raw  # D3 does not depend on the basis
        qz_d3 = e_int(xc, replace(s, basis=big), False) + d3_part
        if xc.startswith("skala"):
            cp = cp_d3 = None
        else:
            e_ab = _pyscf_energy(sym, pos, set(), xc, s, False)
            e_a_cp = _pyscf_energy(
                sym, pos, b, xc, s, False
            )  # A in the full dimer basis
            e_b_cp = _pyscf_energy(sym, pos, a, xc, s, False)
            cp = (e_ab - e_a_cp - e_b_cp) * KCAL_PER_HARTREE
            cp_d3 = cp + d3_part
        rows[xc] = {
            "raw": raw,
            "cp": cp,
            "raw_d3": raw_d3,
            "cp_d3": cp_d3,
            "qzvp_raw_d3": qz_d3,
            "d3": d3_part,
        }
        fmt = lambda v: f"{v:8.2f}" if v is not None else f"{'n/a':>8}"  # noqa: E731
        study.say(
            f"  {xc:<12}{raw:8.2f}{fmt(cp)}{raw_d3:8.2f}{fmt(cp_d3)}{qz_d3:9.2f}{d3_part:9.2f}"
        )
    study.say(
        "  CP: Boys-Bernardi counterpoise via ghost atoms (n/a for Skala: no SCF convergence)."
    )
    study.say(
        "  QZVP+D3: raw interaction in def2-QZVP plus the D3 part; little BSSE left at QZVP."
    )
    study.results["dimer"] = {"reference_kcal": ref_kcal, "rows": rows}


# --------------------------------------------------------------------------- basis / speed


def basis(s: Settings, study: Study) -> None:
    """Optimized geometry, energy and time vs basis set, with and without density fitting."""
    study.say(
        f"== Basis sets and density fitting, {s.xc} (D3={s.dispersion}), opt only =="
    )
    study.say(
        f"  {'basis':<11}{'DF':>4}{'E (Ha)':>15}{'O-H A':>9}{'HOH deg':>9}{'steps':>7}"
        f"{'s/step':>8}{'total s':>9}"
    )
    rows = []
    for bas in ["def2-svp", "def2-tzvp", "def2-qzvp"]:
        for df in (False, True):
            sb = replace(s, basis=bas, density_fit=df)
            atoms = load_water(sb)
            t = time.time()
            steps = _relax(
                atoms, sb, study.dir / f"opt_{bas}_{'df' if df else 'nodf'}.log"
            )
            e = atoms.get_potential_energy() / HARTREE_EV
            g = geometry_summary(atoms)
            dt = time.time() - t
            rows.append(
                {
                    "basis": bas,
                    "density_fit": df,
                    "energy_Ha": e,
                    **g,
                    "steps": steps,
                    "time_s": dt,
                }
            )
            study.say(
                f"  {bas:<11}{'yes' if df else 'no':>4}{e:15.8f}{g['r_OH1_A']:9.4f}"
                f"{g['angle_HOH_deg']:9.2f}{steps:7d}{dt / (steps + 1):8.1f}{dt:9.1f}"
            )
    study.say(f"  experiment{'':>20}{ref.R_OH:9.4f}{ref.ANGLE_HOH:9.2f}")
    study.say(
        "  s/step counts the starting point too (steps + 1 energy+force evaluations)."
    )
    study.results["basis"] = rows


# --------------------------------------------------------------------------- solvation


SOLVENTS = [
    "water",
    "dmso",
    "acetonitrile",
    "methanol",
    "chloroform",
    "toluene",
    "hexane",
]


def solvation(s: Settings, study: Study) -> None:
    """Electrostatic solvation energy of H2O in several PCM solvents, plus relaxation in water."""
    from .solvent import dielectric

    study.say(f"== Solvation (IEF-PCM), {s.xc}/{s.basis}, D3={s.dispersion} ==")
    gas = load_water(s)
    _relax(gas, s, study.dir / "opt_gas.log")
    e_gas = gas.get_potential_energy()
    mu_gas = float(np.linalg.norm(gas.get_dipole_moment()) / Debye)
    study.say(f"  gas: E {e_gas / HARTREE_EV:.8f} Ha, dipole {mu_gas:.3f} D")
    study.say(f"\n  {'solvent':<14}{'eps':>8}{'dE_solv kcal/mol':>18}{'dipole D':>10}")
    rows = {}
    for name in SOLVENTS:
        sv = replace(s, solvent=name)
        a = gas.copy()
        a.calc = make_calculator(sv)
        de = (a.get_potential_energy() - e_gas) * KCAL_PER_EV
        mu = float(np.linalg.norm(a.get_dipole_moment()) / Debye)
        rows[name] = {"eps": dielectric(name), "dE_solv_kcal": de, "dipole_D": mu}
        study.say(f"  {name:<14}{dielectric(name):8.2f}{de:18.2f}{mu:10.3f}")

    # Relax in water: the solute polarizes and the geometry adjusts
    sw = replace(s, solvent="water")
    wat = gas.copy()
    wat.calc = make_calculator(sw)
    _relax(wat, sw, study.dir / "opt_water.log")
    de_relaxed = (wat.get_potential_energy() - e_gas) * KCAL_PER_EV
    g_gas, g_wat = geometry_summary(gas), geometry_summary(wat)
    study.say(
        f"\n  water, relaxed: dE_solv {de_relaxed:.2f} kcal/mol "
        f"(exp. hydration free energy {ref.HYDRATION_DG_KCAL:.2f})"
    )
    study.say(
        f"  geometry gas -> water: O-H {g_gas['r_OH1_A']:.4f} -> {g_wat['r_OH1_A']:.4f} A, "
        f"HOH {g_gas['angle_HOH_deg']:.2f} -> {g_wat['angle_HOH_deg']:.2f} deg"
    )
    study.say(
        "  PCM is electrostatics only: no cavity/dispersion terms and no explicit H-bonds."
    )
    study.results["solvation"] = {
        "E_gas_Ha": e_gas / HARTREE_EV,
        "dipole_gas_D": mu_gas,
        "solvents": rows,
        "water_relaxed_dE_kcal": de_relaxed,
        "geometry_gas": g_gas,
        "geometry_water": g_wat,
    }


# --------------------------------------------------------------------------- CLI

STUDIES = {
    "functionals": functionals,
    "energetics": energetics,
    "dimer": dimer,
    "basis": basis,
    "solvation": solvation,
}


def main(argv: list[str] | None = None) -> Study:
    p = argparse.ArgumentParser(
        prog="python -m flows.water.studies",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("study", choices=list(STUDIES))
    d = Settings()
    p.add_argument("--xc", default=d.xc, help="Skala model for energetics/dimer/basis")
    p.add_argument(
        "--basis", default=d.basis, help="basis for functionals/energetics/dimer"
    )
    p.add_argument("--device", default=d.device, choices=["cpu", "cuda"])
    p.add_argument("--no-dispersion", action="store_true")
    p.add_argument("--out", type=Path, default=Path("runs"))
    a = p.parse_args(argv)
    s = Settings(
        xc=a.xc, basis=a.basis, device=a.device, dispersion=not a.no_dispersion
    )
    study = Study(a.study, s, a.out)
    try:
        STUDIES[a.study](s, study)
    finally:
        study.finish()
    return study


if __name__ == "__main__":
    main()
