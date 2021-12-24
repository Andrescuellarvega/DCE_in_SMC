import numpy as np
import scipy.constants as constants

hp = constants.h  # Planck constant, (Js)
hbar = constants.hbar  # Reduced Planck constant hp/2*pi (Js)
kb = constants.Boltzmann  # Boltzmann constant (J/K)
ec = constants.elementary_charge  # Elementary charge, (Coulombs)
phi0 = hp/(2 * ec)  # Magnetic flux quantum (Js/C=J/A)

# System Constants
ic = 1.25 * 10 ** (-6)  # Critical current of Josephson junctions in SQUID (A)
vcpw = 1.2*10**8  # Propagation velocity in CPW  (m/s), fig4
z0 = 55  # Characteristic impedance of CPW (Ohms), fig4
E_j = (ic * phi0) / (2 * np.pi)  # Josephson energy of SQUID, fig4
E0_j = 1.3 * E_j  # Tunable Josephson energy for static applied magnetic flux, fig4
dE_j = E0_j / 4  # Amplitude of tunable Josephson energy for weak harmonic drive, fig4
L0 = z0 / vcpw  # Inductance of CPW per unit length,  (henry/m =V*s/A*m)
L0_eff = ((phi0 / (2 * np.pi)) ** 2) * (1 / (E0_j * L0))  # Effective length (m), Eq.(18)
dL_eff = L0_eff / 4  # Modulation amplitude of effective length (m)
omega_d = (2*np.pi*18.6)*10**9  # Drive angular frequency, where omega_d = 2*pi*f, f= 18.6 GHz, fig4