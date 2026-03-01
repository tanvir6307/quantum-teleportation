"""
Leakage noise model for transmon qubits.

Transmon qubits are weakly anharmonic oscillators. During gate operations,
population can leak from the computational subspace {|0⟩, |1⟩} to higher
levels (primarily |2⟩). This leakage is not corrected by standard error
correction and constitutes a non-trivial error source.

We model leakage as a depolarizing-like channel since Qiskit noise models
operate in the qubit subspace. The leaked population is effectively lost
and replaced by a mixed state.

References:
    - Rol et al. (2020) PRL
    - Wood & Gambetta (2018) PRA
    - Chen et al. (2016) PRL
"""

import numpy as np
from qiskit_aer.noise.errors import depolarizing_error


def build_leakage_error_single(leakage_rate):
    """
    Create a single-qubit leakage error modeled as depolarizing noise.

    The leakage probability is mapped to a depolarizing channel that
    approximates the effect of population leaving the computational subspace.

    Parameters
    ----------
    leakage_rate : float
        Probability of leakage per gate (typically ~0.0005 for single-qubit).

    Returns
    -------
    QuantumError
        Qiskit quantum error object.
    """
    if leakage_rate <= 0:
        return None
    # Leakage effectively depolarizes the qubit
    return depolarizing_error(leakage_rate, 1)


def build_leakage_error_two_qubit(leakage_rate):
    """
    Create a two-qubit leakage error for CNOT gates.

    CNOT gates have higher leakage due to strong drive pulses
    (typically ~0.005 per CNOT).

    Parameters
    ----------
    leakage_rate : float
        Probability of leakage per CNOT gate.

    Returns
    -------
    QuantumError
        Qiskit two-qubit quantum error object.
    """
    if leakage_rate <= 0:
        return None
    return depolarizing_error(leakage_rate, 2)


def leakage_infidelity_contribution(num_cnots, leakage_rate_cnot,
                                     num_single_gates=0, leakage_rate_single=0):
    """
    Estimate total leakage contribution to infidelity.

    Parameters
    ----------
    num_cnots : int
        Number of CNOT gates in the circuit.
    leakage_rate_cnot : float
        Leakage rate per CNOT.
    num_single_gates : int
        Number of single-qubit gates.
    leakage_rate_single : float
        Leakage rate per single-qubit gate.

    Returns
    -------
    float
        Estimated infidelity contribution from leakage.
    """
    return (num_cnots * leakage_rate_cnot +
            num_single_gates * leakage_rate_single)
