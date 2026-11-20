"""Experimental reference data for gas-phase H2O, used only for side-by-side reports.

Sources: equilibrium geometry and harmonic frequencies from the NIST CCCBDB
(experimental, H2O); fundamentals from the NIST Chemistry WebBook; dipole from
the CRC Handbook of Chemistry and Physics.
"""

R_OH = 0.9578  # Angstrom, r_e
ANGLE_HOH = 104.48  # degrees, theta_e
DIPOLE_DEBYE = 1.855

# (label, harmonic omega_e, observed fundamental) in cm^-1, ordered low -> high
MODES = [
    ("bend (v2)", 1648.5, 1594.7),
    ("sym. stretch (v1)", 3832.2, 3657.1),
    ("asym. stretch (v3)", 3942.5, 3755.9),
]

# Energetics (experiment / thermochemical tables)
ATOMIZATION_D0_KCAL = 219.36  # H2O -> O(3P) + 2 H, 0 K; ATcT 917.80 kJ/mol
BDE_D0_KCAL = 117.64  # H2O -> H + OH, 0 K; 41145.9 cm-1 (Maksyutenko et al. 2006)
IE_ADIABATIC_EV = 12.621  # H2O -> H2O+ (X 2B1), NIST WebBook
IE_VERTICAL_EV = (
    12.62  # 1b1 band maximum; the 0-0 line is the most intense, so ~ adiabatic
)
OH_HARMONIC_CM = 3737.8  # OH X 2Pi, omega_e (Huber & Herzberg)
