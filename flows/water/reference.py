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
