"""
Fidelity calculation utilities.

State fidelity, process fidelity, and related metrics.
"""

import numpy as np
from qiskit.quantum_info import state_fidelity, DensityMatrix, Statevector


def calculate_state_fidelity(rho_ideal, rho_actual):
    """
    Calculate state fidelity F(rho_ideal, rho_actual).

    F = (Tr[sqrt(sqrt(rho_ideal) * rho_actual * sqrt(rho_ideal))])^2

    Parameters
    ----------
    rho_ideal : DensityMatrix or Statevector
    rho_actual : DensityMatrix

    Returns
    -------
    float
        Fidelity in [0, 1].
    """
    return state_fidelity(rho_ideal, rho_actual)


def calculate_purity(rho):
    """
    Calculate purity Tr(rho^2).

    Purity = 1 for pure states, 1/d for maximally mixed d-dimensional state.
    """
    if isinstance(rho, DensityMatrix):
        mat = rho.data
    else:
        mat = np.array(rho)
    return float(np.real(np.trace(mat @ mat)))


def calculate_process_fidelity_from_states(input_states, output_states_ideal,
                                             output_states_actual):
    """
    Estimate average process fidelity from input-output state pairs.

    F_process ≈ (1/N) Σ F(rho_ideal_i, rho_actual_i)

    This is a rough estimate; proper process tomography would be more accurate.

    Parameters
    ----------
    input_states : list
        Input state vectors.
    output_states_ideal : list[DensityMatrix]
        Ideal output density matrices.
    output_states_actual : list[DensityMatrix]
        Actual output density matrices.

    Returns
    -------
    float
        Estimated process fidelity.
    """
    fidelities = []
    for rho_ideal, rho_actual in zip(output_states_ideal, output_states_actual):
        fidelities.append(state_fidelity(rho_ideal, rho_actual))
    return np.mean(fidelities)


def classical_fidelity_limit():
    """
    Return the classical fidelity limit for qubit teleportation.

    F_classical = 2/3 for single-qubit state transfer via classical means.
    Any F > 2/3 certifies quantum teleportation.

    Returns
    -------
    float
        2/3
    """
    return 2.0 / 3.0


def is_quantum_teleportation(fidelity):
    """Check if fidelity exceeds the classical limit."""
    return fidelity > classical_fidelity_limit()
