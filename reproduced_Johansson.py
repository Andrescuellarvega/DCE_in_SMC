"""
Program performs analytical and numerical calculation for photon flux density emitted by an oscillating superconducting
quantum interference device (SQUID)-terminated coplanar waveguide. Produces plot.

Calculation reproduced from "Dynamical Casimir effect in superconducting microwave circuits" by Johansson et. al. (2010)
(see figure 8).

https://journals.aps.org/pra/abstract/10.1103/PhysRevA.82.052509

"""


import j_constants
import matplotlib.pyplot as plt
import numpy as np

"""
---------------------------------------------------------------------------------------------------------------------
 FUNCTION DEFINITIONS
---------------------------------------------------------------------------------------------------------------------
"""


def n_out_analytic(n_bins, temp):
    """
    Performs analytical calculation for DCE radiation.

    :param n_bins = number of bins to partition the frequency range into,
    :param temp = temperature for the thermal output, in Kelvin.
    :return: n_out_dce = array containing DCE photon flux density as a function of frequency,
    :return: n_out_thermal = array containing thermal  ' ' ',
    :return: n_out_total = sum of n_out_dce and n_out_thermal.

    """

    def n_in_analytic(omega, temp):
        if temp == 0:
            n_in = 0  # Zero temperature state
        else:
            with np.errstate(divide='ignore'):  # Ignore the error when dividing by zero
                n_in = np.ma.masked_invalid \
                    (1 / (np.exp(
                        (j_constants.hbar * np.abs(omega)) / (j_constants.kb * temp)) - 1))  # Thermal initial state
            return n_in

    omega = np.linspace(j_constants.omega_d / n_bins, 2 * j_constants.omega_d, n_bins, endpoint=False)

    n_out_dce = ((j_constants.dL_eff / j_constants.vcpw) ** 2) * omega * (j_constants.omega_d - omega) * np.heaviside(j_constants.omega_d - omega, 1)

    n_out_thermal = n_in_analytic(omega, temp) + ((j_constants.dL_eff / j_constants.vcpw) ** 2) \
                    * (omega * np.abs(omega - j_constants.omega_d)) * n_in_analytic(np.abs(omega - j_constants.omega_d), temp)

    n_out_total = n_out_dce + n_out_thermal

    return n_out_dce, n_out_thermal, n_out_total


def n_out_numerical(num_freqs, n_sidebands, N_bins, temp):
    """
    Performs numerical calculation for DCE radiation.

    :param num_freqs = 2-D array of frequencies for different sidebands used in numerical calculation,
    :param n_bins = number of bins to partition the frequency range into,
    :param n_sidebands = number of sidebands to use in numerical calculation,
    :param temp = temperature for the thermal output, in Kelvin.
    :return: n_out_therm_num = array containing thermal photon flux density as a function of frequency,
    :return: n_out_total_num = sum of n_out_dce and n_out_thermal.

    """

    def delta(x, y):
        if x == y:
            d = 1
        else:
            d = 0
        return d

    def g(individual_freq, n, m):
        g = delta(n, m) + 0.5 * (j_constants.dE_j / j_constants.E0_j) * np.sqrt(np.abs(individual_freq[m - 1] / individual_freq[n - 1])) * (
                    delta(n, m + 1) + delta(n, m - 1))
        return g

    def coefficients(frequency_grid, k):

        ww = frequency_grid[k - 1, :]  # Index runs 0 < k < N_bins-2, effective 1 < k < N_bins-1
        m_out = np.empty((2 * n_sidebands + 1, 2 * n_sidebands + 1), dtype=complex)
        m_in = np.empty((2 * n_sidebands + 1, 2 * n_sidebands + 1), dtype=complex)

        # Populate M_out, M_in matrices
        for m in range(1, 2 * n_sidebands + 2):  # Indexes run 0 < m,n < 8, effectively 1 < m,n < 9
            for n in range(1, 2 * n_sidebands + 2):
                m_out[m - 1, n - 1] = -g(ww, n, m) + np.complex(0, 1) * (ww[m - 1] / j_constants.vcpw) * j_constants.L0_eff * delta(n, m)
                m_in[m - 1, n - 1] = g(ww, n, m) + np.complex(0, 1) * (ww[m - 1] / j_constants.vcpw) * j_constants.L0_eff * delta(n, m)

        a_matrix = np.matmul(np.linalg.inv(m_out), m_in)

        u = np.zeros((2 * n_sidebands + 1))
        v = np.zeros((2 * n_sidebands + 1))

        for m in range(1, 2 * n_sidebands + 2):
            # upper portion, N_omega in mathematica
            v[m - 1] = np.real(np.conj(a_matrix[n_sidebands - 1, m - 1]) * a_matrix[n_sidebands - 1, m - 1])

            # lower portion, N_omega + 1 in mathematica
            u[m - 1] = np.real(np.conj(a_matrix[n_sidebands, m - 1]) * a_matrix[n_sidebands, m - 1])

        return u, v

    def n_in_numerical(frequency, temperature):
        if temperature == 0:
            n_in = 0
        else:
            n_in = 1 / (np.exp(j_constants.hbar * np.abs(frequency) / (j_constants.kb * temperature)) - 1)

        return n_in

    #  -----------------------------------------------------------------------------------------------------------------
    # ACTUAL CALCULATION
    #  -----------------------------------------------------------------------------------------------------------------
    noutthermnum_lower = np.zeros(N_bins - 1)
    noutdcenum_lower = np.zeros(N_bins - 1)
    noutthermnum_upper = np.zeros(N_bins - 1)
    noutdcenum_upper = np.zeros(N_bins - 1)

    sum_array_u1 = np.zeros(2 * n_sidebands + 1)
    sum_array_u2 = np.zeros(2 * n_sidebands + 1)
    sum_array_v1 = np.zeros(2 * n_sidebands + 1)
    sum_array_v2 = np.zeros(2 * n_sidebands + 1)

    for k in range(1, N_bins):  # 1 < k < 99

        u, v = coefficients(num_freqs, k)  # Populate coefficients for (0 < omega_k < omega_d),(omega_d < omega_k < 2*omega_d)

        for m in range(1, 2 * n_sidebands + 2):                               # Populate thermal part of the output
            sum_array_u1[m-1] = n_in_numerical(num_freqs[k - 1, m - 1], temp) * u[m - 1]    # Runs 0<m<8, effectively 1<m<9
            sum_array_v1[m-1] = n_in_numerical(num_freqs[k - 1, m - 1], temp) * v[m - 1]    # Runs 0<m<8, effectively 1<m<9

        noutthermnum_lower[k - 1] = np.sum(sum_array_u1)
        noutthermnum_upper[k - 1] = np.sum(sum_array_v1)

        for m in range(n_sidebands + 2, 2 * n_sidebands + 2):     # Populate DCE part of output
            sum_array_u2[m-1] = u[m-1]              # Runs 5<m<8 which is effectively 6<m<9
            sum_array_v2[m-1] = v[m-1]              # Runs 5<m<8 which is effectively 6<m<9

        noutdcenum_lower[k-1] = np.sum(sum_array_u2)
        noutdcenum_upper[k-1] = np.sum(sum_array_v2)

    # Joining upper and lower arrays into single

    n_out_total_num = np.concatenate((noutdcenum_lower + noutthermnum_lower, noutdcenum_upper + noutthermnum_upper))
    n_out_therm_num = np.concatenate((noutthermnum_lower, noutthermnum_upper))

    return n_out_total_num, n_out_therm_num


"""
---------------------------------------------------------------------------------------------------------------------
 ANALYTICAL RUN
---------------------------------------------------------------------------------------------------------------------
"""

N_bins = 10000
temperature1 = 0.025 # In Kelvin
temperature2 = 0.05

# For 25 mK
(n_an_dce, n_an_thermal25, n_an_total25) = n_out_analytic(N_bins, temperature1)
# For 50 mK
(n_an_dce, n_an_thermal50, n_an_total50) = n_out_analytic(N_bins, temperature2)

frequency_grid_an = np.linspace(j_constants.omega_d / N_bins, 2 * j_constants.omega_d, N_bins, endpoint=False) / j_constants.omega_d  # For plotting


"""
---------------------------------------------------------------------------------------------------------------------
 NUMERICAL RUN
---------------------------------------------------------------------------------------------------------------------
"""

N_sidebands = 5

numerical_freqs = np.empty((N_bins - 1, 2 * N_sidebands + 1))  # Populate www, N_bins - 1 to avoid singularities

for k in range(1, N_bins):
    for m in range(1, 2 * N_sidebands + 2):  # m = 1,...,2*N_omega+1,  9 elements for 4 sidebands
        numerical_freqs[k - 1, m - 1] = k * (j_constants.omega_d / N_bins) + (N_sidebands + 1 - m) * j_constants.omega_d

frequency_grid_num = np.concatenate((numerical_freqs[:, N_sidebands] / j_constants.omega_d, numerical_freqs[:, N_sidebands - 1] / j_constants.omega_d))

# For T = 0 K
n_num_dce, n_num_thermal0 = n_out_numerical(numerical_freqs, N_sidebands, N_bins, 0)
# For 25 mK
n_num_total25, n_num_thermal25 = n_out_numerical(numerical_freqs, N_sidebands, N_bins, temperature1)
# For 50 mK
n_num_total50, n_num_thermal50 = n_out_numerical(numerical_freqs, N_sidebands, N_bins, temperature2)


"""
---------------------------------------------------------------------------------------------------------------------
 PLOTS
---------------------------------------------------------------------------------------------------------------------
"""


fig, ax = plt.subplots(figsize=(9, 7), dpi=600)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.xlim(0, 2)
plt.ylim(0, 4*10**(-3))
plt.xlabel("$ \omega / \omega_d $",fontsize=26)
plt.yticks(np.arange(0, 5*10**(-3), step=10**(-3)))
plt.ylabel("$ n_{out}(\omega)$",fontsize=26)
#plt.title("Output-field photon flux density.")

#an_dce, = ax.plot(frequency_grid_an, n_an_dce, 'b:', label='T = 0 K')
#an_thermal25, = ax.plot(frequency_grid_an, n_an_thermal25, 'r--',  label='Thermal, T = 25 mK')
#an_thermal50, = ax.plot(frequency_grid_an, n_an_thermal50, 'r',  label='Thermal, T = 50 mK')

num_dce = ax.plot(frequency_grid_num, n_num_dce, 'b--', label='DCE')
num_therm25 = ax.plot(frequency_grid_num, n_num_thermal25, 'r--', label='Thermal, T = 25mK')
num_therm50 = ax.plot(frequency_grid_num, n_num_thermal50, 'r', label='Thermal, T = 50mK')

#an_total25, = ax.plot(frequency_grid_an, n_an_total25, 'b--',  label='DCE + Thermal (Analytical) T = 25 mK')
num_total25 = ax.plot(frequency_grid_num, n_num_total25, 'g--', label='DCE + Thermal, T = 25 mK')

#an_total50, = ax.plot(frequency_grid_an, n_an_total50, 'b', label='DCE + Thermal(Analytical), T = 50 mK')
num_total50 = ax.plot(frequency_grid_num, n_num_total50, 'g', label='DCE + Thermal, T = 50 mK')

plt.legend()
plt.tight_layout()
plt.savefig("reproduced_Johansson.png")
