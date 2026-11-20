"""Building blocks shared by the water flows: settings, calculator, geometry, output."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from skala.ase import Skala

HARTREE_EV = 27.211386245988
EV_TO_CM = 8065.543937


@dataclass
class Settings:
    """Everything that defines a calculation. Saved next to every result."""

    xc: str = "skala-1.1"
    basis: str = "def2-tzvp"
    device: str = "cpu"  # "cpu" or "cuda"
    dispersion: bool = True  # D3 correction, the Skala calculator's default
    fmax: float = 0.005  # eV/A, optimization convergence on the largest force
    fd_step: float = 0.005  # A, finite-difference displacement for frequencies
    temperature: float = 298.15  # K
    pressure: float = 101325.0  # Pa
    geometry: str | None = None  # optional .xyz to start from instead of ideal H2O


def make_calculator(s: Settings) -> Skala:
    return Skala(xc=s.xc, basis=s.basis, device=s.device, with_dftd3=s.dispersion)


def load_water(s: Settings) -> Atoms:
    atoms = read(s.geometry) if s.geometry else molecule("H2O")
    if sorted(atoms.get_chemical_symbols()) != ["H", "H", "O"]:
        raise ValueError(
            f"Expected one water molecule, got {atoms.get_chemical_formula()}"
        )
    # Put O first so atom indices are always O=0, H=1, H=2.
    order = np.argsort(
        [0 if sym == "O" else 1 for sym in atoms.get_chemical_symbols()], kind="stable"
    )
    atoms = atoms[order]
    atoms.calc = make_calculator(s)
    return atoms


def geometry_summary(atoms: Atoms) -> dict:
    return {
        "r_OH1_A": float(atoms.get_distance(0, 1)),
        "r_OH2_A": float(atoms.get_distance(0, 2)),
        "angle_HOH_deg": float(atoms.get_angle(1, 0, 2)),
    }


@dataclass
class Run:
    """One output folder: runs/<timestamp>_<flows>/ with results.json and report.txt."""

    settings: Settings
    flows: list[str]
    root: Path = Path("runs")
    results: dict = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = self.root / f"{stamp}_{'-'.join(self.flows)}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.t0 = time.time()

    def say(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def save_xyz(self, name: str, atoms: Atoms, comment: str) -> Path:
        path = self.dir / f"{name}.xyz"
        write(path, atoms, comment=comment)
        return path

    def finish(self) -> None:
        self.results["wall_time_s"] = round(time.time() - self.t0, 1)
        payload = {
            "settings": asdict(self.settings),
            "flows": self.flows,
            **self.results,
        }
        (self.dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
        self.say(f"\nWall time {self.results['wall_time_s']} s. Results in {self.dir}/")
        (self.dir / "report.txt").write_text("\n".join(self.lines) + "\n")
