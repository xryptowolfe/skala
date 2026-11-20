# Water flows: Skala in one command

Ready-made workflows for a single H₂O molecule with the Skala-1.1 neural
exchange-correlation functional: single point, geometry optimization, harmonic
frequencies with IR intensities, ideal-gas thermochemistry, and energy scans.

## Setup (CPU, macOS Apple Silicon or Linux)

```bash
bash flows/setup_env.sh
```

This creates `.venv/` with Python 3.12, CPU-only PyTorch and the `skala` package
from this repository. The first calculation downloads the skala-1.1 checkpoint
from Hugging Face (about once; it is cached in `~/.cache/huggingface`).

On a GPU, use the Colab notebook
[`water_flows_colab.ipynb`](water_flows_colab.ipynb) or install `skala-cuda12x`
and add `--device cuda` to any command.

## Usage

Run from the repository root:

```bash
.venv/bin/python -m flows.water sp                       # energy, dipole, forces
.venv/bin/python -m flows.water opt                      # relax the geometry
.venv/bin/python -m flows.water opt freq thermo          # the usual "opt freq" job
.venv/bin/python -m flows.water scan --geometry runs/<run>/optimized.xyz
.venv/bin/python -m flows.water scan --scan-coord stretch --geometry runs/<run>/optimized.xyz
```

Flows always run in the order `sp → opt → freq → thermo → scan`, each on the
geometry the previous one left, so `opt freq` gives frequencies at the
optimized structure. `thermo` runs `freq` itself if needed.

| Option | Default | Meaning |
|---|---|---|
| `--xc` | `skala-1.1` | Skala model |
| `--basis` | `def2-tzvp` | basis set (Skala was trained and tested with def2-TZVP) |
| `--device` | `cpu` | `cuda` uses GPU4PySCF |
| `--no-dispersion` | off | turn off the D3 dispersion correction |
| `--fmax` | `0.005` | optimization threshold on the largest force, eV/Å |
| `--fd-step` | `0.005` | finite-difference displacement for frequencies, Å |
| `--temperature`, `--pressure` | `298.15`, `101325` | K, Pa, for `thermo` |
| `--geometry` | ideal H₂O | start from an `.xyz` file (atoms in any order) |
| `--scan-coord`, `--scan-points` | `angle`, `9` | rigid scan of H–O–H ±15° or both O–H ±0.1 Å |
| `--out` | `runs/` | parent folder for result folders |

## Output

Each command makes `runs/<date-time>_<flows>/` containing:

- `report.txt` – the printed report
- `results.json` – all numbers plus the settings used
- `start.xyz`, `optimized.xyz` – geometries
- `opt.traj`, `opt.log` – optimization steps (open with `ase gui opt.traj`)
- `vib/` – cached finite-difference forces and dipoles
- `scan_<coord>.png` – energy curve, for `scan`

## Reference results

Skala-1.1/def2-TZVP with D3, measured with `opt freq thermo` on an Apple
Silicon MacBook Air (CPU, about 2.5 minutes):

| Quantity | Skala-1.1 | Experiment |
|---|---:|---:|
| O–H (Å) | 0.9596 | 0.9578 (r_e) |
| H–O–H (°) | 103.93 | 104.48 (θ_e) |
| Bend (cm⁻¹) | 1680.4 | 1648.5 harmonic / 1594.7 fundamental |
| Symmetric stretch (cm⁻¹) | 3865.4 | 3832.2 / 3657.1 |
| Asymmetric stretch (cm⁻¹) | 3972.3 | 3942.5 / 3755.9 |
| ZPE (eV) | 0.590 | |
| S at 298 K (J mol⁻¹ K⁻¹) | 188.6 | 188.8 |
| Dipole (D, at the ideal starting geometry) | 2.08 | 1.855 |

Compare Skala's frequencies with the **harmonic** experimental values; the
observed fundamentals are lower because of anharmonicity. The dipole is
overestimated partly because def2-TZVP has no diffuse functions.

## How it works

- **Optimization:** ASE BFGS on the Skala energy and analytic forces.
- **Frequencies:** central finite differences of the analytic forces (3 atoms ×
  3 directions × 2 = 18 force calculations plus the reference point). Skala has
  no analytic Hessian. Translations and rotations are projected out of the
  mass-weighted Hessian, as Gaussian does: the DFT integration grid is not
  exactly rotation-invariant, which otherwise leaves the six "zero" modes at
  roughly 100–200 cm⁻¹. The vibrational frequencies change by less than
  0.1 cm⁻¹ from the projection and by less than 0.5 cm⁻¹ between 0.005 and
  0.01 Å steps.
- **IR intensities:** from finite differences of the dipole moment (km/mol).
- **Thermochemistry:** ASE `IdealGasThermo` (ideal gas, rigid rotor, harmonic
  oscillator; symmetry number 2, singlet).
- **Scans:** rigid (no relaxation of the other coordinates); the minimum comes
  from a quartic fit, because the curve is anharmonic over the scanned range.

## Test

```bash
.venv/bin/python -m pytest flows/tests -v
```

Runs `opt freq thermo` with def2-SVP (about 3 minutes on a laptop CPU) and
checks the geometry, the three modes, the ZPE and the entropy fall in physical
ranges. The `water-flows` GitHub Actions workflow runs the same test.
