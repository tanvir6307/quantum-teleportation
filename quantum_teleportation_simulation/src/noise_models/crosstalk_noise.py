"""
Cross-talk noise model: ZZ coupling between neighboring qubits.

In transmon qubits, residual ZZ coupling causes unwanted conditional phase
accumulation between idle qubits during gate operations on neighboring qubits.
This is a dominant source of correlated errors.

References:
    - Malekakhlagh et al. (2020) PRX
    - Mundada et al. (2019) PRApplied
"""

import numpy as np
from qiskit.quantum_info import Operator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import coherent_unitary_error


def zz_unitary(theta):
    """
    Create the ZZ interaction unitary: exp(-i * theta/2 * ZZ).

    Parameters
    ----------
    theta : float
        Rotation angle = 2*pi*zz_rate*gate_time.

    Returns
    -------
    np.ndarray
        4x4 unitary matrix.
    """
    U = np.diag([
        np.exp(-1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(-1j * theta / 2),
    ])
    return U


def build_crosstalk_noise(zz_coupling_Hz, gate_duration_cnot_ns, coupling_map, num_qubits):
    """
    Build a noise model for ZZ cross-talk during CNOT gates.

    During a CNOT on qubits (c, t), spectator qubits coupled to c or t
    experience a ZZ phase accumulation. We model this as a coherent error
    on the spectator-active qubit pair.

    Parameters
    ----------
    zz_coupling_Hz : dict
        ZZ coupling rates in Hz, keyed by "(q1,q2)" strings.
    gate_duration_cnot_ns : float
        CNOT gate duration in nanoseconds.
    coupling_map : list[tuple]
        Device coupling map.
    num_qubits : int
        Number of qubits.

    Returns
    -------
    dict
        Dictionary mapping CNOT qubit pairs to lists of (spectator, theta) pairs.
        This is used by the composite noise model.
    """
    t_cnot_s = gate_duration_cnot_ns * 1e-9
    crosstalk_info = {}

    for pair in coupling_map:
        c, t = pair
        spectators = []
        for zz_key, zz_rate in zz_coupling_Hz.items():
            # Parse key like "(0,1)"
            q1, q2 = _parse_qubit_pair(zz_key)
            if q1 is None:
                continue
            # Check if this ZZ pair involves one of the active qubits
            # but also involves a spectator
            if c in (q1, q2) or t in (q1, q2):
                spectator = None
                if q1 not in (c, t) and q1 < num_qubits:
                    spectator = q1
                elif q2 not in (c, t) and q2 < num_qubits:
                    spectator = q2
                if spectator is not None:
                    theta = 2 * np.pi * abs(zz_rate) * t_cnot_s
                    spectators.append((spectator, theta))

        crosstalk_info[(c, t)] = spectators

    return crosstalk_info


def _parse_qubit_pair(key_str):
    """Parse a qubit pair string like '(0,1)' into (0, 1)."""
    try:
        cleaned = key_str.strip("()")
        parts = cleaned.split(",")
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, IndexError):
        return None, None


def crosstalk_phase_error(theta):
    """
    Convert a ZZ phase angle into process infidelity.

    The ZZ unitary is U = diag(e^{-iθ/2}, e^{iθ/2}, e^{iθ/2}, e^{-iθ/2}).
    Process infidelity vs identity on the spectator qubit:
        1 - F_process = sin²(θ/2)

    References:
        - Nielsen & Chuang, Eq. (9.85)
        - Malekakhlagh et al. (2020) PRX
    """
    return min(np.sin(theta / 2) ** 2, 1.0)
