"""
Collective dephasing noise model.

In multi-qubit systems, qubits can experience correlated dephasing
from shared noise sources (e.g., global magnetic field fluctuations,
shared control lines). This creates correlated errors distinct from
independent per-qubit dephasing.

For the teleportation protocol, collective dephasing primarily
affects the Bell pair qubits when they experience correlated
frequency fluctuations.

References:
    - Clemens et al. (2004) PRA
    - Chirolli & Burkard (2008) Advances in Physics
"""

import numpy as np
from qiskit_aer.noise.errors import phase_damping_error, depolarizing_error


def collective_dephasing_rate(T2_q1, T2_q2, correlation=0.3):
    """
    Estimate collective dephasing rate for two qubits.

    With correlation parameter rho, the collective dephasing
    introduces correlated ZZ-like phase errors.

    Parameters
    ----------
    T2_q1, T2_q2 : float
        Individual qubit T2 times.
    correlation : float
        Noise correlation coefficient (0 = independent, 1 = fully correlated).

    Returns
    -------
    float
        Additional dephasing probability from collective effects.
    """
    # Geometric mean of dephasing rates
    gamma_avg = 0.5 * (1.0 / T2_q1 + 1.0 / T2_q2)
    # Collective contribution scales with correlation
    gamma_collective = correlation * gamma_avg
    return gamma_collective


def build_collective_dephasing_error(T2_q1, T2_q2, duration,
                                      correlation=0.3):
    """
    Build a two-qubit collective dephasing error.

    Approximated as correlated phase damping on both qubits.

    Parameters
    ----------
    T2_q1, T2_q2 : float
        T2 times for the two qubits.
    duration : float
        Time window in seconds.
    correlation : float
        Correlation coefficient.

    Returns
    -------
    QuantumError or None
    """
    gamma_c = collective_dephasing_rate(T2_q1, T2_q2, correlation)
    p = 1 - np.exp(-gamma_c * duration)
    if p < 1e-8:
        return None
    # Model as two-qubit depolarizing with reduced probability
    return depolarizing_error(p * correlation, 2)


def collective_dephasing_infidelity(T2_values, duration, correlation=0.3):
    """
    Estimate collective dephasing contribution to protocol infidelity.

    Parameters
    ----------
    T2_values : list[float]
        T2 for each qubit involved.
    duration : float
        Protocol duration in seconds.
    correlation : float
        Noise correlation.

    Returns
    -------
    float
        Estimated infidelity contribution.
    """
    if len(T2_values) < 2:
        return 0.0
    gamma_c = collective_dephasing_rate(T2_values[0], T2_values[1], correlation)
    return 1 - np.exp(-gamma_c * duration)
