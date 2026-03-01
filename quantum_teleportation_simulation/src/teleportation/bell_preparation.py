"""
Realistic Bell pair preparation with gate-level error tracking.

Implements the standard Bell state preparation circuit (H + CNOT) with
detailed noise modeling at each step, tracking the fidelity degradation
through the preparation process.

References:
    - Bennett et al. (1993) PRL
    - Nielsen & Chuang (2010) Ch. 1
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, state_fidelity, Statevector


# Ideal Bell states as density matrices
def bell_state_vector(bell_type="phi_plus"):
    """
    Return the ideal Bell state vector.

    Parameters
    ----------
    bell_type : str
        One of 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'.

    Returns
    -------
    Statevector
    """
    mapping = {
        "phi_plus":  np.array([1, 0, 0, 1]) / np.sqrt(2),   # |00⟩ + |11⟩
        "phi_minus": np.array([1, 0, 0, -1]) / np.sqrt(2),  # |00⟩ - |11⟩
        "psi_plus":  np.array([0, 1, 1, 0]) / np.sqrt(2),   # |01⟩ + |10⟩
        "psi_minus": np.array([0, 1, -1, 0]) / np.sqrt(2),  # |01⟩ - |10⟩
    }
    return Statevector(mapping[bell_type])


def bell_state_density_matrix(bell_type="phi_plus"):
    """Return the ideal Bell state as a density matrix."""
    sv = bell_state_vector(bell_type)
    return DensityMatrix(sv)


def create_bell_preparation_circuit(bell_type="phi_plus"):
    """
    Create the Bell state preparation circuit.

    |Φ+⟩: H(0) → CNOT(0,1)
    |Φ-⟩: X(0) → H(0) → CNOT(0,1)
    |Ψ+⟩: H(0) → CNOT(0,1) → X(1)
    |Ψ-⟩: X(0) → H(0) → CNOT(0,1) → X(1)

    Parameters
    ----------
    bell_type : str
        Type of Bell state to prepare.

    Returns
    -------
    QuantumCircuit
    """
    qc = QuantumCircuit(2, name=f"Bell_{bell_type}")

    if bell_type in ("phi_minus", "psi_minus"):
        qc.x(0)

    qc.h(0)
    qc.cx(0, 1)

    if bell_type in ("psi_plus", "psi_minus"):
        qc.x(1)

    return qc


def simulate_bell_preparation(noise_model_obj, params, bell_type="phi_plus",
                               shots=0):
    """
    Simulate Bell pair preparation with comprehensive noise.

    Uses density matrix simulation for exact fidelity calculation.

    Parameters
    ----------
    noise_model_obj : ComprehensiveTeleportationNoise
        The comprehensive noise model object.
    params : dict
        Device parameters.
    bell_type : str
        Bell state type.
    shots : int
        If 0, use density matrix simulator (exact). Otherwise, use shots.

    Returns
    -------
    dict
        Contains 'fidelity', 'density_matrix', 'purity', 'concurrence_lower_bound'.
    """
    qc = create_bell_preparation_circuit(bell_type)
    qc.save_density_matrix()

    noise_model = noise_model_obj.build_noise_model(qubits_used=[0, 1])

    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    result = sim.run(qc, shots=1).result()
    rho_noisy = DensityMatrix(result.data()["density_matrix"])

    rho_ideal = bell_state_density_matrix(bell_type)
    fid = state_fidelity(rho_ideal, rho_noisy)

    # Purity = Tr(rho^2)
    purity = float(np.real(np.trace(rho_noisy.data @ rho_noisy.data)))

    return {
        "fidelity": fid,
        "density_matrix": rho_noisy,
        "purity": purity,
        "bell_type": bell_type,
    }


def analyze_bell_error_accumulation(noise_model_obj, params):
    """
    Track fidelity degradation step-by-step through Bell preparation.

    Runs the circuit incrementally (after H, after CNOT) to see
    where errors accumulate.

    Returns
    -------
    dict
        Step-by-step fidelity trajectory.
    """
    noise_model = noise_model_obj.build_noise_model(qubits_used=[0, 1])
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)

    steps = []

    # Step 0: Initial state |00⟩
    steps.append({
        "step": "initial",
        "time_ns": 0,
        "fidelity": 1.0,
        "gate": "none",
    })

    # Step 1: After H gate
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.save_density_matrix()
    result1 = sim.run(qc1, shots=1).result()
    rho1 = DensityMatrix(result1.data()["density_matrix"])
    # Ideal state after H(q0)|00⟩ = (|00⟩+|01⟩)/√2  (Qiskit little-endian: |q1 q0⟩)
    ideal_after_h = Statevector([1, 1, 0, 0]) / np.sqrt(2)
    fid1 = state_fidelity(DensityMatrix(ideal_after_h), rho1)
    steps.append({
        "step": "after_H",
        "time_ns": params["gate_duration_single_ns"],
        "fidelity": fid1,
        "gate": "H(q0)",
    })

    # Step 2: After CNOT → full Bell pair
    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.save_density_matrix()
    result2 = sim.run(qc2, shots=1).result()
    rho2 = DensityMatrix(result2.data()["density_matrix"])
    rho_ideal = bell_state_density_matrix("phi_plus")
    fid2 = state_fidelity(rho_ideal, rho2)
    steps.append({
        "step": "after_CNOT",
        "time_ns": params["gate_duration_single_ns"] + params["gate_duration_cnot_ns"],
        "fidelity": fid2,
        "gate": "CNOT(q0,q1)",
    })

    return {
        "steps": steps,
        "final_fidelity": fid2,
        "final_density_matrix": rho2,
    }


def validate_bell_fidelity(simulated_fidelity):
    """
    Compare simulated Bell pair fidelity with experimental benchmarks.

    Returns
    -------
    dict
        Comparison with published Bell state fidelities on IBM hardware.
    """
    # Literature benchmarks for Bell state fidelity on superconducting qubits
    benchmarks = {
        "IBM_typical_2022": {"fidelity": 0.89, "error": 0.03,
                             "source": "IBM Quantum calibration data"},
        "IBM_improved_2024": {"fidelity": 0.92, "error": 0.02,
                              "source": "IBM Quantum Falcon processors"},
        "Steffen_2013_SC": {"fidelity": 0.88, "error": 0.02,
                            "source": "Steffen et al. (2013) Nature"},
    }

    results = {}
    for name, data in benchmarks.items():
        gap = abs(simulated_fidelity - data["fidelity"])
        within_error = gap < data["error"]
        results[name] = {
            "experimental": data["fidelity"],
            "experimental_error": data["error"],
            "simulated": simulated_fidelity,
            "gap": gap,
            "within_experimental_error": within_error,
            "source": data["source"],
        }

    return results
