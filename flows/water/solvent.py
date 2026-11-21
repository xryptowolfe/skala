"""Implicit solvent for the water flows: Skala inside PySCF's PCM continuum.

PySCF's ``.PCM()`` wraps any Kohn-Sham object, including Skala's ``SkalaKS``, and also
provides analytic nuclear gradients for the wrapped object. ``SkalaPCM`` exposes this
as an ASE calculator, so every flow (sp, opt, freq, thermo, scan) can run in solvent.

The continuum gives the electrostatic part of solvation only: cavity formation and
dispersion with the solvent (the non-electrostatic terms in SMD) are not included.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Debye, Hartree

# Static dielectric constants at 298 K (the values Gaussian and PySCF use for SMD).
DIELECTRIC = {
    "water": 78.3553,
    "dmso": 46.826,
    "acetonitrile": 35.688,
    "methanol": 32.613,
    "ethanol": 24.852,
    "acetone": 20.493,
    "dichloromethane": 8.93,
    "thf": 7.4257,
    "chloroform": 4.7113,
    "toluene": 2.3741,
    "benzene": 2.2706,
    "hexane": 1.8819,
}

MODELS = ("IEF-PCM", "C-PCM", "COSMO", "SS(V)PE")


def dielectric(solvent: str) -> float:
    """Dielectric constant for a solvent name, or a number given as text (e.g. "10.0").

    Args:
        solvent: A key of ``DIELECTRIC`` (case-insensitive) or a positive number.

    Returns:
        The static dielectric constant.

    Raises:
        ValueError: If the name is unknown and not a number greater than 1.
    """
    key = solvent.lower()
    if key in DIELECTRIC:
        return DIELECTRIC[key]
    try:
        eps = float(solvent)
    except ValueError:
        raise ValueError(
            f"Unknown solvent {solvent!r}; use one of {', '.join(DIELECTRIC)} or a number"
        ) from None
    if eps <= 1.0:
        raise ValueError("A dielectric constant must be greater than 1")
    return eps


class SkalaPCM(Calculator):
    """ASE calculator: Skala (or any functional SkalaKS accepts) in a PCM continuum.

    Args:
        xc: Functional name, e.g. "skala-1.1" or "pbe".
        basis: Basis set name.
        solvent: Solvent name or dielectric constant (see ``dielectric``).
        model: PCM variant, one of ``MODELS``.
        dispersion: Add the D3 correction.
        density_fit: Use density fitting for the Coulomb term.
        charge: Total charge.
        multiplicity: Spin multiplicity (2S + 1); > 1 uses UKS.
        device: "cpu" or "cuda".
    """

    implemented_properties = ["energy", "forces", "dipole"]

    def __init__(
        self,
        xc: str = "skala-1.1",
        basis: str = "def2-tzvp",
        solvent: str = "water",
        model: str = "IEF-PCM",
        dispersion: bool = True,
        density_fit: bool = False,
        charge: int = 0,
        multiplicity: int = 1,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if model not in MODELS:
            raise ValueError(f"PCM model must be one of {MODELS}")
        self.cfg = dict(
            xc=xc,
            basis=basis,
            eps=dielectric(solvent),
            model=model,
            dispersion=dispersion,
            density_fit=density_fit,
            charge=charge,
            spin=multiplicity - 1,
            device=device,
        )

    def _build(self, atoms: Atoms):
        from pyscf import gto

        c = self.cfg
        mol = gto.M(
            atom=[
                (s, tuple(p))
                for s, p in zip(
                    atoms.get_chemical_symbols(), atoms.positions, strict=True
                )
            ],
            basis=c["basis"],
            charge=c["charge"],
            spin=c["spin"],
            verbose=0,
        )
        if c["device"] == "cuda":
            from skala.gpu4pyscf import SkalaKS
        else:
            from skala.pyscf import SkalaKS
        ks = SkalaKS(
            mol,
            xc=c["xc"],
            with_dftd3=c["dispersion"],
            with_density_fit=c["density_fit"],
        ).PCM()
        ks.with_solvent.method = c["model"]
        ks.with_solvent.eps = c["eps"]
        return ks

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties or ["energy"], system_changes)
        from skala.pyscf.retry import retry_scf

        ks, _ = retry_scf(self._build(self.atoms))
        if not ks.converged:
            raise RuntimeError("SCF in solvent did not converge")
        grad = np.asarray(ks.nuc_grad_method().kernel())
        dip = np.asarray(ks.dip_moment(unit="debye", verbose=0))
        self.results = {
            "energy": float(ks.e_tot) * Hartree,
            "forces": -grad * Hartree / Bohr,
            "dipole": dip * Debye,
        }
