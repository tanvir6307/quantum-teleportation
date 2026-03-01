"""
Coherent control errors: systematic over/under-rotation and axis misalignment.

Real quantum gates are implemented via microwave/flux pulses that are
never perfectly calibrated. This leads to systematic (coherent) errors
that are distinct from stochastic (incoherent) noise.

These errors include:
- Over-rotation: gate angle theta → theta*(1 + epsilon)
- Under-rotation: similar but with negative epsilon
- Axis tilt: rotation axis slightly off from ideal

Coherent errors are typically small (~0.1%) but can accumulate
constructively, unlike depolarizing noise.

References:
    - Barends et al. (2014) Nature
    - McKay et al. (2017) PRA
"""

import numpy as np
from qiskit_aer.noise.errors import coherent_unitary_error


def over_rotation_unitary_rz(theta, epsilon):
    """
    Rz gate with over-rotation: Rz(theta*(1+epsilon)).

    Parameters
    ----------
    theta : float
        Ideal rotation angle.
    epsilon : float
        Fractional over-rotation (e.g., 0.001 = 0.1%).

    Returns
    -------
    np.ndarray
        2x2 unitary matrix.
    """
    angle = theta * (1 + epsilon)
    return np.array([
        [np.exp(-1j * angle / 2), 0],
        [0, np.exp(1j * angle / 2)]
    ])


def over_rotation_unitary_rx(theta, epsilon):
    """Rx gate with over-rotation."""
    angle = theta * (1 + epsilon)
    return np.array([
        [np.cos(angle / 2), -1j * np.sin(angle / 2)],
        [-1j * np.sin(angle / 2), np.cos(angle / 2)]
    ])


def coherent_error_probability(epsilon):
    """
    Convert coherent over-rotation parameter to effective error probability.

    For small epsilon, the error probability is approximately epsilon^2.
    This is useful for error budget analysis.

    Parameters
    ----------
    epsilon : float
        Fractional over-rotation.

    Returns
    -------
    float
        Effective error probability.
    """
    return min(epsilon**2, 1.0)


def build_coherent_error_single_qubit(epsilon):
    """
    Build a coherent over-rotation error for single-qubit gates.

    Uses a unitary error channel U = Rz(epsilon * pi).

    Parameters
    ----------
    epsilon : float
        Over-rotation fraction.

    Returns
    -------
    QuantumError
    """
    if abs(epsilon) < 1e-10:
        return None
    # Small rotation around Z axis
    U = over_rotation_unitary_rz(np.pi, epsilon)
    return coherent_unitary_error(U)


def build_coherent_error_cnot(epsilon):
    """
    Build a coherent error for CNOT gates.

    Models systematic calibration error as a small Z-rotation on the target
    qubit after the CNOT.

    Parameters
    ----------
    epsilon : float
        Over-rotation fraction.

    Returns
    -------
    QuantumError
    """
    if abs(epsilon) < 1e-10:
        return None
    # Two-qubit coherent error: identity on control, small rotation on target
    U_target = over_rotation_unitary_rz(np.pi, epsilon)
    I2 = np.eye(2)
    U_full = np.kron(I2, U_target)
    return coherent_unitary_error(U_full)
