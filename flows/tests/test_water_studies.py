"""Fast checks for flows.water.studies (the full studies take minutes to tens of minutes)."""

import pytest
from ase import Atoms
from flows.water import studies
from flows.water.core import HARTREE_EV, Settings, make_calculator

pytestmark = pytest.mark.timeout(900)


def test_all_studies_registered():
    assert set(studies.STUDIES) == {
        "functionals",
        "energetics",
        "dimer",
        "basis",
        "solvation",
    }


def test_hydrogen_atom_open_shell():
    # Exact nonrelativistic H atom energy is -0.5 Ha; a good functional is close.
    h = Atoms("H", positions=[[0, 0, 0]])
    h.calc = make_calculator(Settings(basis="def2-svp"), multiplicity=2)
    assert h.get_potential_energy() / HARTREE_EV == pytest.approx(-0.5, abs=0.005)


def test_counterpoise_ghost_monomer_is_lower():
    # A monomer gains a little energy from the partner's basis functions (BSSE),
    # so the ghost-augmented monomer must not be higher than the bare one.
    from ase.data.s22 import create_s22_system

    d = create_s22_system("Water_dimer")
    sym, pos = d.get_chemical_symbols(), d.get_positions()
    s = Settings(basis="def2-svp")
    bare = studies._pyscf_energy(sym[:3], pos[:3], set(), "pbe", s, False)
    ghost = studies._pyscf_energy(sym, pos, {3, 4, 5}, "pbe", s, False)
    assert ghost <= bare + 1e-8
    assert bare - ghost < 0.01  # BSSE is small (< ~6 kcal/mol)


def test_solvent_lowers_energy_of_polar_molecule():
    # A polar molecule is stabilized by a dielectric continuum; more so in water than hexane.
    from ase.build import molecule

    def energy(solvent):
        w = molecule("H2O")
        w.calc = make_calculator(Settings(basis="def2-svp", xc="pbe", solvent=solvent))
        return w.get_potential_energy()

    gas, hexane, water = energy(None), energy("hexane"), energy("water")
    assert water < hexane < gas
    assert -0.5 < (water - gas) < -0.1  # eV; about -5 kcal/mol


def test_dielectric_lookup():
    from flows.water.solvent import dielectric

    assert dielectric("Water") == pytest.approx(78.3553)
    assert dielectric("10") == 10.0
    with pytest.raises(ValueError):
        dielectric("not-a-solvent")
