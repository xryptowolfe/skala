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
[`water_flows_colab.ipynb`](water_flows_colab.ipynb) or install `skala-cuda12x` (it also works on CUDA 13 drivers; `skala-cuda13x` fails on a T4)
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

The same run on a Colab Tesla T4 (`--device cuda`) took 140 s and agreed with the
CPU run to within 1×10⁻⁷ Ha in energy and 0.3 cm⁻¹ in frequency. For a
3-atom molecule the GPU is barely faster than a laptop CPU; it pays off for
larger molecules.

Compare Skala's frequencies with the **harmonic** experimental values; the
observed fundamentals are lower because of anharmonicity. The dipole is
overestimated partly because def2-TZVP has no diffuse functions.

## Studies

`python -m flows.water.studies <study>` runs larger comparisons built on the flows.
Each writes `runs/<date-time>_study-<name>/` with `report.txt`, `results.json` and
the sub-runs. Numbers below: Apple Silicon MacBook Air, CPU, def2-TZVP.

### `functionals`: Skala versions vs traditional functionals (~25 min)

Optimization and harmonic frequencies with each functional, D3 off for all.
MAE is the mean absolute error of the three frequencies against the
experimental harmonic values.

| Functional | O–H (Å) | H–O–H (°) | Bend | Sym. | Asym. | MAE (cm⁻¹) | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPW92 (LDA) | 0.9716 | 105.25 | 1533.9 | 3704.4 | 3818.5 | 122.1 | 11 |
| PBE | 0.9710 | 104.28 | 1583.4 | 3691.5 | 3799.8 | 116.1 | 23 |
| TPSS | 0.9692 | 104.46 | 1619.3 | 3703.3 | 3809.9 | 96.9 | 45 |
| SCAN | 0.9620 | 104.49 | 1631.9 | 3794.7 | 3899.6 | 32.3 | 24 |
| r2SCAN | 0.9624 | 104.49 | 1625.3 | 3793.6 | 3901.8 | 34.2 | 26 |
| Skala-1.0 | 0.9515 | 105.05 | 1674.3 | 3942.6 | 4050.4 | 81.4 | 236 |
| Skala-1.1 | 0.9596 | 103.93 | 1680.3 | 3865.4 | 3972.5 | 31.6 | 390–660 |
| Experiment | 0.9578 | 104.48 | 1648.5 | 3832.2 | 3942.5 | | |

Skala-1.1 has the best O–H bond length and ties SCAN for frequencies; its
weak spot is the angle (−0.55°). `skala-1.1-rev0` gives the same results as
`skala-1.1` (rev1) to 4×10⁻¹² Ha. On a CPU Skala costs 10–25× a meta-GGA.

### `energetics`: open-shell thermochemistry (~8 min)

O(³P), H, OH and H₂O⁺ run as unrestricted (UKS) calculations through the
calculator's `charge` and `multiplicity` options.

| Quantity | Skala-1.1 | Experiment | Error |
|---|---:|---:|---:|
| Atomization energy D0 (kcal/mol) | 215.62 | 219.36 | −3.74 |
| O–H bond energy D0(H–OH) (kcal/mol) | 116.70 | 117.64 | −0.94 |
| Ionization energy, adiabatic (eV) | 12.72 | 12.62 | +0.10 |
| Ionization energy, vertical (eV) | 12.82 | 12.62 | +0.20 |

ZPE comes from the Skala frequencies. The adiabatic ionization energy is
electronic only, and atomic spin-orbit coupling (~0.2 kcal/mol) is ignored.

### `dimer`: hydrogen bond in (H₂O)₂

Interaction energy of the S22 water dimer at its reference geometry
(reference CCSD(T)/CBS −5.02 kcal/mol), in kcal/mol. Takes ~6 min.

| Functional | raw | CP | raw+D3 | CP+D3 | **QZVP+D3** | D3 part |
|---|---:|---:|---:|---:|---:|---:|
| PBE | −6.07 | −5.34 | −6.48 | −5.75 | **−5.72** | −0.41 |
| r2SCAN | −5.95 | −5.42 | −6.08 | −5.55 | **−5.52** | −0.14 |
| Skala-1.1 | −4.97 | n/a | −5.61 | n/a | **−4.92** | −0.64 |

Columns are def2-TZVP unless noted. CP is the Boys–Bernardi counterpoise
correction for basis-set superposition error, computed with ghost atoms.
Skala's SCF does not converge with ghost atoms, so for Skala the error is
judged instead from the raw interaction in def2-QZVP (QZVP+D3), where little
superposition error remains: for PBE and r2SCAN, QZVP+D3 matches CP+D3 within
0.03 kcal/mol. On that common footing Skala-1.1 with D3 is within
0.1 kcal/mol of the reference, while PBE and r2SCAN overbind by 0.5–0.7.

### `basis`: basis sets and density fitting (~12 min)

| Basis | DF | E (Ha) | O–H (Å) | H–O–H (°) | Time (s) |
|---|---|---:|---:|---:|---:|
| def2-SVP | no | −76.32261479 | 0.9634 | 101.72 | 55 |
| def2-SVP | yes | −76.32270670 | 0.9633 | 101.72 | 84 |
| def2-TZVP | no | −76.42953492 | 0.9596 | 103.93 | 147 |
| def2-TZVP | yes | −76.42967449 | 0.9596 | 103.94 | 92 |
| def2-QZVP | no | −76.43877967 | 0.9575 | 103.70 | 133 |
| def2-QZVP | yes | −76.43894628 | 0.9574 | 103.71 | 178 |

def2-SVP is too small (angle 2.8° low). def2-TZVP and def2-QZVP agree within
0.002 Å and 0.2°, so def2-TZVP is the default. Density fitting moves energies
by 0.1–0.2 mHa and gives no consistent speedup for water: the neural XC
evaluation, not the Coulomb term, dominates the cost. Timings vary by tens of
percent between runs.

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
