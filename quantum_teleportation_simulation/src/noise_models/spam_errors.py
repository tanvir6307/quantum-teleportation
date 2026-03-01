"""
State Preparation and Measurement (SPAM) errors.

SPAM errors include:
1. State preparation errors: thermal population of |1⟩ at initialization
   (qubit not perfectly in |0⟩ due to finite temperature).
2. Measurement errors: readout confusion (misidentifying |0⟩ as |1⟩ or vice versa).

These are typically calibrated independently and are a significant
contributor to overall protocol infidelity.

References:
    - IBM Quantum calibration data
    - Gambetta et al. (2007) PRA
"""

import numpy as np
from qiskit_aer.noise import ReadoutError
from qiskit_aer.noise.errors import pauli_error


def build_state_prep_error(thermal_population):
    """
    Build a state preparation error.

    Models the thermal excitation: with probability `thermal_population`,
    the qubit starts in |1⟩ instead of |0⟩. This is implemented as a
    bit-flip error on the reset/initialization.

    Parameters
    ----------
    thermal_population : float
        Probability of being in |1⟩ after reset (e.g., 0.02).

    Returns
    -------
    QuantumError
        Pauli-X error with probability `thermal_population`.
    """
    if thermal_population <= 0:
        return None
    return pauli_error([("X", thermal_population), ("I", 1 - thermal_population)])


def build_readout_error(error_rate):
    """
    Build a symmetric readout error (confusion matrix).

    Symmetric model: P(read 1 | prepared 0) = P(read 0 | prepared 1) = error_rate.

    Parameters
    ----------
    error_rate : float
        Probability of misclassification.

    Returns
    -------
    ReadoutError
        Qiskit readout error object.
    """
    # Confusion matrix:
    # [[P(0|0), P(1|0)],
    #  [P(0|1), P(1|1)]]
    return ReadoutError(
        [[1 - error_rate, error_rate],
         [error_rate, 1 - error_rate]]
    )


def build_asymmetric_readout_error(p_0_given_1, p_1_given_0):
    """
    Build an asymmetric readout error.

    Parameters
    ----------
    p_0_given_1 : float
        P(measure 0 | state is 1) — false negative.
    p_1_given_0 : float
        P(measure 1 | state is 0) — false positive.

    Returns
    -------
    ReadoutError
    """
    return ReadoutError(
        [[1 - p_1_given_0, p_1_given_0],
         [p_0_given_1, 1 - p_0_given_1]]
    )


def spam_infidelity_contribution(thermal_population, readout_errors, num_measurements=2):
    """
    Estimate total SPAM contribution to infidelity.

    Parameters
    ----------
    thermal_population : float
        State prep error probability.
    readout_errors : list[float]
        Readout error for each measured qubit.
    num_measurements : int
        Number of qubits measured.

    Returns
    -------
    dict
        SPAM error budget breakdown.
    """
    prep_error = thermal_population
    meas_error = sum(readout_errors[:num_measurements]) / num_measurements
    total = prep_error + meas_error  # First-order approximation

    return {
        "state_prep": prep_error,
        "readout_mean": meas_error,
        "readout_per_qubit": readout_errors[:num_measurements],
        "total_spam": total,
    }
