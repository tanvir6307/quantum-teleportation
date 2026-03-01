"""
Quantum teleportation circuit construction.

Implements the standard Bennett et al. (1993) teleportation protocol:
1. Bell pair preparation (qubits 1, 2)
2. Alice's Bell measurement (qubits 0, 1)
3. Classical communication (2 bits)
4. Bob's conditional corrections (qubit 2)

The circuit is constructed with barriers to clearly delineate
protocol phases for noise analysis.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector


def create_teleportation_circuit(input_state=None, with_corrections=True):
    """
    Create the standard quantum teleportation circuit.

    Parameters
    ----------
    input_state : array-like or None
        2-element complex array [alpha, beta] for the state to teleport.
        If None, leaves qubit 0 in |0⟩ (caller should initialize).
    with_corrections : bool
        If True, include Bob's conditional X and Z corrections.
        If False, omit corrections (for deferred measurement).

    Returns
    -------
    QuantumCircuit
        3-qubit teleportation circuit with 2 classical bits.
    """
    qr = QuantumRegister(3, name="q")
    cr = ClassicalRegister(2, name="c")
    qc = QuantumCircuit(qr, cr, name="Teleportation")

    # ── PHASE 0: Prepare input state on qubit 0 ────────────────────
    if input_state is not None:
        input_state = np.array(input_state, dtype=complex)
        norm = np.linalg.norm(input_state)
        if abs(norm - 1.0) > 1e-6:
            input_state = input_state / norm
        _apply_state_prep_gates(qc, 0, input_state)

    qc.barrier(label="Init")

    # ── PHASE 1: Bell pair preparation (qubits 1 & 2) ──────────────
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier(label="Bell")

    # ── PHASE 2: Alice's Bell measurement circuit (qubits 0 & 1) ───
    qc.cx(0, 1)
    qc.h(0)
    qc.barrier(label="Alice")

    # ── PHASE 3: Measurement ───────────────────────────────────────
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.barrier(label="Meas")

    # ── PHASE 4: Bob's conditional corrections (qubit 2) ───────────
    if with_corrections:
        with qc.if_test((cr[1], 1)):
            qc.x(2)
        with qc.if_test((cr[0], 1)):
            qc.z(2)

    return qc


def create_pre_measurement_circuit(input_state=None):
    """
    Create the teleportation circuit up to (but NOT including) measurement.

    This builds:
    1. State preparation on qubit 0
    2. Bell pair preparation on qubits 1, 2
    3. Alice's BSM gates (CNOT(0,1), H(0))

    Measurement, classical communication delay, and Bob's corrections
    are handled analytically by the protocol executor to properly model
    readout errors and idle decoherence.

    Parameters
    ----------
    input_state : array-like or None
        State to teleport.

    Returns
    -------
    QuantumCircuit
        3-qubit circuit ending after Alice's H gate.
    """
    qc = QuantumCircuit(3, name="Teleportation_PreMeas")

    if input_state is not None:
        input_state = np.array(input_state, dtype=complex)
        norm = np.linalg.norm(input_state)
        if abs(norm - 1.0) > 1e-6:
            input_state = input_state / norm
        _apply_state_prep_gates(qc, 0, input_state)

    # Bell pair (qubits 1, 2)
    qc.h(1)
    qc.cx(1, 2)

    # Alice's BSM gates (qubits 0, 1)
    qc.cx(0, 1)
    qc.h(0)

    return qc


def create_teleportation_circuit_deferred(input_state=None):
    """
    Create teleportation circuit using deferred measurement principle.

    Instead of mid-circuit measurements and classical feedback,
    uses quantum-controlled operations. This gives the quantum channel
    fidelity (no readout errors, no classical delay).

    Parameters
    ----------
    input_state : array-like or None
        State to teleport.

    Returns
    -------
    QuantumCircuit
        3-qubit circuit without mid-circuit measurement.
    """
    qc = QuantumCircuit(3, name="Teleportation_Deferred")

    if input_state is not None:
        input_state = np.array(input_state, dtype=complex)
        norm = np.linalg.norm(input_state)
        if abs(norm - 1.0) > 1e-6:
            input_state = input_state / norm
        _apply_state_prep_gates(qc, 0, input_state)

    # Bell pair
    qc.h(1)
    qc.cx(1, 2)

    # Alice's operations
    qc.cx(0, 1)
    qc.h(0)

    # Bob's corrections (controlled, not classically conditioned)
    qc.cx(1, 2)  # X correction controlled by qubit 1
    qc.cz(0, 2)  # Z correction controlled by qubit 0

    return qc


def _apply_state_prep_gates(qc, qubit, state_vector):
    """
    Prepare an arbitrary single-qubit state using Ry and Rz gates.

    |psi> = cos(theta/2)|0> + e^{i*phi}*sin(theta/2)|1>
    => Ry(theta) then Rz(phi)
    """
    alpha = state_vector[0]
    beta = state_vector[1]
    # theta = 2 * arccos(|alpha|)
    theta = 2 * np.arccos(np.clip(np.abs(alpha), 0, 1))
    # phi = arg(beta) - arg(alpha)
    phi = np.angle(beta) - np.angle(alpha)

    if abs(theta) > 1e-10:
        qc.ry(theta, qubit)
    if abs(phi) > 1e-10:
        qc.rz(phi, qubit)


def get_test_states():
    """
    Return the standard set of 6 test states for tomographic validation.

    These span the Bloch sphere:
    |0⟩, |1⟩, |+⟩, |-⟩, |+i⟩, |-i⟩

    Returns
    -------
    list[dict]
        Each dict has 'name', 'vector', 'bloch_coords'.
    """
    return [
        {
            "name": "|0⟩",
            "vector": np.array([1, 0], dtype=complex),
            "bloch_coords": (0, 0, 1),
        },
        {
            "name": "|1⟩",
            "vector": np.array([0, 1], dtype=complex),
            "bloch_coords": (0, 0, -1),
        },
        {
            "name": "|+⟩",
            "vector": np.array([1, 1], dtype=complex) / np.sqrt(2),
            "bloch_coords": (1, 0, 0),
        },
        {
            "name": "|-⟩",
            "vector": np.array([1, -1], dtype=complex) / np.sqrt(2),
            "bloch_coords": (-1, 0, 0),
        },
        {
            "name": "|+i⟩",
            "vector": np.array([1, 1j], dtype=complex) / np.sqrt(2),
            "bloch_coords": (0, 1, 0),
        },
        {
            "name": "|-i⟩",
            "vector": np.array([1, -1j], dtype=complex) / np.sqrt(2),
            "bloch_coords": (0, -1, 0),
        },
    ]
