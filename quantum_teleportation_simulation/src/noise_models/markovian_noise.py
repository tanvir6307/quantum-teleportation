"""
Markovian thermal relaxation noise (T1 amplitude damping, T2 phase damping).

This models the dominant decoherence channel in superconducting transmon qubits.
T1 describes energy relaxation (|1⟩ → |0⟩ decay).
T2 describes total dephasing (pure dephasing + T1 contribution).

References:
    - Nielsen & Chuang (2010), Ch. 8
    - IBM Quantum calibration data
"""

import numpy as np
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import thermal_relaxation_error


def build_thermal_relaxation_noise(
    T1_per_qubit,
    T2_per_qubit,
    gate_duration_single_ns,
    gate_duration_cnot_ns,
    measurement_duration_ns,
    coupling_map,
    num_qubits,
):
    """
    Build a Qiskit NoiseModel with thermal relaxation on every gate.

    Parameters
    ----------
    T1_per_qubit : list[float]
        T1 times in seconds for each qubit.
    T2_per_qubit : list[float]
        T2 times in seconds for each qubit.
    gate_duration_single_ns : float
        Single-qubit gate duration in ns.
    gate_duration_cnot_ns : float
        CNOT gate duration in ns.
    measurement_duration_ns : float
        Measurement duration in ns.
    coupling_map : list[tuple]
        List of (control, target) pairs.
    num_qubits : int
        Total number of qubits.

    Returns
    -------
    NoiseModel
        Qiskit noise model with thermal relaxation errors.
    """
    noise_model = NoiseModel()

    t_single = gate_duration_single_ns * 1e-9  # convert to seconds
    t_cnot = gate_duration_cnot_ns * 1e-9
    t_meas = measurement_duration_ns * 1e-9

    for q in range(num_qubits):
        T1 = T1_per_qubit[q] if q < len(T1_per_qubit) else T1_per_qubit[-1]
        T2 = T2_per_qubit[q] if q < len(T2_per_qubit) else T2_per_qubit[-1]
        # Ensure T2 <= 2*T1 (physical constraint)
        T2 = min(T2, 2 * T1)

        # Single-qubit gate relaxation
        error_single = thermal_relaxation_error(T1, T2, t_single)
        noise_model.add_quantum_error(error_single, ["h", "x", "z", "s", "sdg", "sx", "rz", "ry", "rx", "id"], [q])

        # Measurement relaxation
        error_meas = thermal_relaxation_error(T1, T2, t_meas)
        noise_model.add_quantum_error(error_meas, "measure", [q])

    # Two-qubit gate relaxation
    for pair in coupling_map:
        q0, q1 = pair
        T1_0 = T1_per_qubit[q0] if q0 < len(T1_per_qubit) else T1_per_qubit[-1]
        T2_0 = T2_per_qubit[q0] if q0 < len(T2_per_qubit) else T2_per_qubit[-1]
        T1_1 = T1_per_qubit[q1] if q1 < len(T1_per_qubit) else T1_per_qubit[-1]
        T2_1 = T2_per_qubit[q1] if q1 < len(T2_per_qubit) else T2_per_qubit[-1]
        T2_0 = min(T2_0, 2 * T1_0)
        T2_1 = min(T2_1, 2 * T1_1)

        error_cx = thermal_relaxation_error(T1_0, T2_0, t_cnot).expand(
            thermal_relaxation_error(T1_1, T2_1, t_cnot)
        )
        noise_model.add_quantum_error(error_cx, "cx", [q0, q1])

    return noise_model


def t1_decay_probability(duration, T1):
    """Compute amplitude damping probability gamma = 1 - exp(-t/T1)."""
    return 1 - np.exp(-duration / T1)


def t2_dephasing_rate(duration, T1, T2):
    """Compute pure dephasing rate gamma_phi."""
    rate_phi = 1.0 / T2 - 1.0 / (2.0 * T1)
    if rate_phi < 0:
        rate_phi = 0
    return 1 - np.exp(-duration * rate_phi)


def simulate_t1_decay(times, T1):
    """
    Simulate T1 relaxation: population of |1⟩ vs time.

    Returns array of excited state populations.
    """
    return np.exp(-np.array(times) / T1)
