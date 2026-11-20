"""The individual water flows. Each takes (atoms, settings, run) and adds to run.results."""

from __future__ import annotations

import numpy as np
from ase import units
from ase.optimize import BFGS
from ase.thermochemistry import IdealGasThermo
from ase.units import Debye
from ase.vibrations import Infrared

from . import reference as ref
from .core import EV_TO_CM, HARTREE_EV, Run, Settings, geometry_summary, make_calculator


def _geometry_lines(run: Run, g: dict) -> None:
    run.say(
        f"  O-H  {g['r_OH1_A']:.4f}  {g['r_OH2_A']:.4f} A   (exp. r_e {ref.R_OH:.4f})"
    )
    run.say(
        f"  H-O-H {g['angle_HOH_deg']:.2f} deg            (exp. theta_e {ref.ANGLE_HOH:.2f})"
    )


def single_point(atoms, s: Settings, run: Run) -> None:
    """Energy, dipole and forces at the current geometry."""
    e = atoms.get_potential_energy()
    mu = np.linalg.norm(atoms.get_dipole_moment()) / Debye
    fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    g = geometry_summary(atoms)
    run.say(f"\n== Single point ({s.xc}/{s.basis}) ==")
    run.say(f"  E      {e / HARTREE_EV:.8f} Ha   ({e:.6f} eV)")
    run.say(f"  dipole {mu:.3f} D               (exp. {ref.DIPOLE_DEBYE:.3f})")
    run.say(f"  max |F| {fmax:.4f} eV/A")
    _geometry_lines(run, g)
    run.results["single_point"] = {
        "energy_Ha": e / HARTREE_EV,
        "dipole_D": mu,
        "fmax_eV_A": fmax,
        **g,
    }


def optimize(atoms, s: Settings, run: Run) -> None:
    """Relax the geometry until the largest force is below s.fmax."""
    run.say(f"\n== Geometry optimization (fmax {s.fmax} eV/A) ==")
    e0 = atoms.get_potential_energy()
    opt = BFGS(
        atoms, trajectory=str(run.dir / "opt.traj"), logfile=str(run.dir / "opt.log")
    )
    converged = bool(opt.run(fmax=s.fmax, steps=200))
    e = atoms.get_potential_energy()
    g = geometry_summary(atoms)
    run.say(f"  converged {converged} in {opt.nsteps} steps")
    run.say(f"  E      {e / HARTREE_EV:.8f} Ha   (dE {(e - e0) * 1000:.2f} meV)")
    _geometry_lines(run, g)
    run.save_xyz("optimized", atoms, f"{s.xc}/{s.basis} E={e / HARTREE_EV:.8f} Ha")
    run.results["optimization"] = {
        "converged": converged,
        "steps": opt.nsteps,
        "energy_Ha": e / HARTREE_EV,
        **g,
    }
    if not converged:
        raise RuntimeError(
            "Optimization did not converge; frequencies would not be meaningful."
        )


def frequencies(atoms, s: Settings, run: Run) -> None:
    """Harmonic frequencies and IR intensities by central finite differences."""
    run.say(f"\n== Harmonic frequencies (finite differences, step {s.fd_step} A) ==")
    fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    if fmax > 10 * s.fmax:
        run.say(
            f"  WARNING: max |F| = {fmax:.3f} eV/A, not a stationary point. Run 'opt' first."
        )
    ir = Infrared(atoms, name=str(run.dir / "vib"), delta=s.fd_step, nfree=2)
    ir.run()
    ir.read()
    raw = ir.get_energies()  # unprojected, eV, complex for imaginary modes
    km_mol = ir.intensities * ir.intensity_prefactor("km/mol")[0]
    raw_rest = float(np.sort(np.abs(raw))[:6].max() * EV_TO_CM)

    # Project translations and rotations out of the mass-weighted Hessian, as
    # Gaussian does. The DFT grid is not exactly rotation-invariant, so without
    # this the 6 "zero" modes can reach ~100-200 cm-1.
    hnu, vecs = _projected_modes(atoms, ir.H, ir.im)
    vib = np.argsort(np.abs(hnu))[6:]
    vib = vib[np.argsort(hnu[vib].real - np.abs(hnu[vib].imag))]
    overlap = np.abs(vecs[:, vib].T @ ir.modes.T)  # match to ASE modes for IR intensity

    modes = []
    run.say(
        f"  {'mode':<20}{'Skala cm-1':>11}{'exp. harm.':>11}{'exp. fund.':>11}{'IR km/mol':>11}"
    )
    for k, ((label, w_harm, w_fund), i) in enumerate(zip(ref.MODES, vib, strict=True)):
        e = hnu[i]
        w = -abs(e.imag) * EV_TO_CM if abs(e.imag) > 1e-8 else e.real * EV_TO_CM
        inten = float(km_mol[np.argmax(overlap[k])])
        run.say(f"  {label:<20}{w:11.1f}{w_harm:11.1f}{w_fund:11.1f}{inten:11.1f}")
        modes.append({"label": label, "freq_cm": float(w), "ir_km_mol": inten})
    n_imag = sum(m["freq_cm"] < 0 for m in modes)
    zpe = 0.5 * sum(m["freq_cm"] for m in modes if m["freq_cm"] > 0) / EV_TO_CM
    run.say(
        f"  imaginary modes {n_imag}   ZPE {zpe:.4f} eV ({zpe / HARTREE_EV:.6f} Ha)"
    )
    run.say(
        f"  trans./rot. residual before projection {raw_rest:.1f} cm-1 (DFT grid; removed)"
    )
    run.results["frequencies"] = {
        "modes": modes,
        "n_imaginary": n_imag,
        "zpe_eV": zpe,
        "trans_rot_residual_unprojected_cm": raw_rest,
    }
    run._vib_energies = [hnu[i] for i in vib]


def _projected_modes(atoms, hessian, inv_sqrt_mass):
    """Eigen-decompose the mass-weighted Hessian with translations/rotations projected out.

    Returns (hnu in eV as complex, mass-weighted eigenvectors as columns).
    """
    m = atoms.get_masses()
    r = atoms.get_positions() - atoms.get_center_of_mass()
    sq = np.sqrt(m)
    basis = []
    for k in range(3):  # translations
        t = np.zeros((len(m), 3))
        t[:, k] = sq
        basis.append(t.ravel())
    for k in range(3):  # rotations about x, y, z
        axis = np.eye(3)[k]
        basis.append((np.cross(axis, r) * sq[:, None]).ravel())
    q, _ = np.linalg.qr(np.array(basis).T)
    proj = np.eye(len(q)) - q @ q.T
    hm = inv_sqrt_mass[:, None] * hessian * inv_sqrt_mass
    omega2, vecs = np.linalg.eigh(proj @ hm @ proj)
    conv = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
    return conv * omega2.astype(complex) ** 0.5, vecs


def thermo(atoms, s: Settings, run: Run) -> None:
    """Ideal-gas, rigid-rotor, harmonic-oscillator thermochemistry (needs frequencies)."""
    if not hasattr(run, "_vib_energies"):
        frequencies(atoms, s, run)
    e = atoms.get_potential_energy()
    th = IdealGasThermo(
        vib_energies=run._vib_energies,
        geometry="nonlinear",
        potentialenergy=e,
        atoms=atoms,
        symmetrynumber=2,  # C2v
        spin=0,
    )
    T, P = s.temperature, s.pressure
    h = th.get_enthalpy(T, verbose=False)
    sv = th.get_entropy(T, P, verbose=False)
    g = th.get_gibbs_energy(T, P, verbose=False)
    run.say(f"\n== Thermochemistry (ideal gas, {T} K, {P / 1000:.3f} kPa) ==")
    run.say(f"  H   {h / HARTREE_EV:.8f} Ha   (H - E = {(h - e) * 1000:.1f} meV)")
    run.say(
        f"  S   {sv * 1000:.4f} meV/K = {sv * 96485.332:.2f} J/(mol K)   (exp. 188.8)"
    )
    run.say(f"  G   {g / HARTREE_EV:.8f} Ha   (G - E = {(g - e) * 1000:.1f} meV)")
    run.results["thermo"] = {
        "T_K": T,
        "P_Pa": P,
        "H_Ha": h / HARTREE_EV,
        "S_J_molK": sv * 96485.332,
        "G_Ha": g / HARTREE_EV,
    }


def scan(atoms, s: Settings, run: Run, coord: str = "angle", points: int = 9) -> None:
    """Rigid scan of the H-O-H angle (+-15 deg) or symmetric O-H stretch (+-0.1 A)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g0 = geometry_summary(atoms)
    if coord == "angle":
        values = np.linspace(g0["angle_HOH_deg"] - 15, g0["angle_HOH_deg"] + 15, points)
        unit = "deg"
    else:
        r0 = 0.5 * (g0["r_OH1_A"] + g0["r_OH2_A"])
        values = np.linspace(r0 - 0.1, r0 + 0.1, points)
        unit = "A"
    run.say(f"\n== Rigid scan of {coord} ({points} points) ==")
    energies = []
    for v in values:
        a = atoms.copy()
        a.calc = make_calculator(s)
        if coord == "angle":
            a.set_angle(1, 0, 2, v, indices=[2])
        else:
            a.set_distance(0, 1, v, fix=0)
            a.set_distance(0, 2, v, fix=0)
        energies.append(a.get_potential_energy())
        run.say(f"  {v:9.4f} {unit}   E {energies[-1] / HARTREE_EV:.8f} Ha")
    rel = (np.array(energies) - min(energies)) * 1000
    # Quartic fit: the curve is anharmonic over this range, so a parabola would
    # shift the minimum outward (noticeably for the O-H stretch).
    fit = np.polyfit(values, rel, 4)
    xs = np.linspace(values[0], values[-1], 2001)
    v_min = float(xs[np.argmin(np.polyval(fit, xs))])
    run.say(
        f"  fitted minimum at {v_min:.4f} {unit} (rigid scan: other coordinates held fixed)"
    )
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(values, rel, "o", label=s.xc)
    ax.plot(xs, np.polyval(fit, xs), "-", lw=1, label="quartic fit")
    ax.set_xlabel(
        f"H-O-H angle ({unit})" if coord == "angle" else f"O-H distance ({unit})"
    )
    ax.set_ylabel("relative energy (meV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run.dir / f"scan_{coord}.png", dpi=150)
    run.results[f"scan_{coord}"] = {
        "values": values.tolist(),
        "energies_Ha": [x / HARTREE_EV for x in energies],
        "fit_minimum": float(v_min),
    }
