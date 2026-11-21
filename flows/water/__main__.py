"""Command line for the water flows.

Examples (from the repository root):
    python -m flows.water sp                     # single point at the ideal geometry
    python -m flows.water opt freq --solvent water   # in implicit water (PCM)
    python -m flows.water opt freq thermo        # the usual "opt freq" job
    python -m flows.water opt scan --scan-coord stretch
    python -m flows.water opt freq --device cuda # same on a GPU (needs gpu4pyscf)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import flows
from .core import Run, Settings, load_water

ORDER = ["sp", "opt", "freq", "thermo", "scan"]


def main(argv: list[str] | None = None) -> Run:
    p = argparse.ArgumentParser(
        prog="python -m flows.water",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "flows",
        nargs="+",
        choices=ORDER,
        help="what to run; always executed in this order: " + " ".join(ORDER),
    )
    d = Settings()
    p.add_argument("--xc", default=d.xc, help=f"Skala model (default {d.xc})")
    p.add_argument("--basis", default=d.basis, help=f"basis set (default {d.basis})")
    p.add_argument("--device", default=d.device, choices=["cpu", "cuda"])
    p.add_argument(
        "--no-dispersion", action="store_true", help="turn off the D3 correction"
    )
    p.add_argument(
        "--fmax",
        type=float,
        default=d.fmax,
        help=f"opt threshold, eV/A (default {d.fmax})",
    )
    p.add_argument(
        "--fd-step",
        type=float,
        default=d.fd_step,
        help=f"freq displacement, A (default {d.fd_step})",
    )
    p.add_argument("--temperature", type=float, default=d.temperature)
    p.add_argument("--pressure", type=float, default=d.pressure, help="Pa")
    p.add_argument(
        "--geometry", help="start from this .xyz instead of the ideal water geometry"
    )
    p.add_argument(
        "--solvent",
        help="implicit solvent (PCM): water, acetonitrile, chloroform, ... or a dielectric",
    )
    p.add_argument(
        "--solvent-model",
        default="IEF-PCM",
        choices=["IEF-PCM", "C-PCM", "COSMO", "SS(V)PE"],
    )
    p.add_argument("--scan-coord", choices=["angle", "stretch"], default="angle")
    p.add_argument("--scan-points", type=int, default=9)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs"),
        help="parent folder for results (default runs/)",
    )
    a = p.parse_args(argv)

    s = Settings(
        xc=a.xc,
        basis=a.basis,
        device=a.device,
        dispersion=not a.no_dispersion,
        fmax=a.fmax,
        fd_step=a.fd_step,
        temperature=a.temperature,
        pressure=a.pressure,
        geometry=a.geometry,
        solvent=a.solvent,
        solvent_model=a.solvent_model,
    )
    chosen = [f for f in ORDER if f in a.flows]
    run = Run(settings=s, flows=chosen, root=a.out)
    atoms = load_water(s)
    run.say(
        f"Skala water flows: {' -> '.join(chosen)}   [{s.xc}/{s.basis}, {s.device}, D3={s.dispersion}, "
        f"{'gas' if not s.solvent else s.solvent_model + ' ' + s.solvent}]"
    )
    if s.solvent:
        run.say(
            "Note: the 'exp.' columns are gas-phase experiment. PCM adds only the electrostatic "
            "part of solvation (no hydrogen bonds to explicit solvent molecules)."
        )
    run.save_xyz("start", atoms, "starting geometry")

    steps = {
        "sp": lambda: flows.single_point(atoms, s, run),
        "opt": lambda: flows.optimize(atoms, s, run),
        "freq": lambda: flows.frequencies(atoms, s, run),
        "thermo": lambda: flows.thermo(atoms, s, run),
        "scan": lambda: flows.scan(atoms, s, run, a.scan_coord, a.scan_points),
    }
    try:
        for f in chosen:
            steps[f]()
    finally:
        run.finish()
    return run


if __name__ == "__main__":
    main()
