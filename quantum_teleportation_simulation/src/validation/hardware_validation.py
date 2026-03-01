"""
Hardware validation module for quantum teleportation.

Executes the teleportation protocol on real IBM Quantum hardware
and compares results with simulation predictions.

Approach:
- Build teleportation circuit with mid-circuit measurement + corrections
- State tomography: measure Bob's qubit in X, Y, Z bases (3 circuits per state)
- Reconstruct density matrix from measurement statistics
- Compute fidelity against ideal states
- Compare with simulation fidelities

Compatible with IBM Quantum Runtime (qiskit-ibm-runtime >= 0.40).
"""

import numpy as np
import time
import warnings
import os

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import state_fidelity, DensityMatrix


def _apply_state_prep_gates(qc, qubit, state_vector):
    """Apply Ry(theta) * Rz(phi) to prepare |psi> = alpha|0> + beta|1>."""
    alpha, beta = state_vector[0], state_vector[1]
    theta = 2 * np.arccos(np.clip(np.abs(alpha), -1, 1))
    if abs(beta) > 1e-10:
        phi = np.angle(beta) - np.angle(alpha)
    else:
        phi = 0.0
    if abs(theta) > 1e-10:
        qc.ry(theta, qubit)
    if abs(phi) > 1e-10:
        qc.rz(phi, qubit)


def build_teleportation_tomography_circuits(input_state, basis='Z'):
    """
    Build a teleportation circuit with tomography measurement on Bob's qubit.

    Parameters
    ----------
    input_state : np.ndarray
        2-element state vector to teleport.
    basis : str
        Measurement basis for Bob's qubit: 'X', 'Y', or 'Z'.

    Returns
    -------
    QuantumCircuit
        Complete circuit with 3 classical bits (2 for Alice, 1 for Bob).
    """
    qr = QuantumRegister(3, name="q")
    cr_alice = ClassicalRegister(2, name="alice")
    cr_bob = ClassicalRegister(1, name="bob")
    qc = QuantumCircuit(qr, cr_alice, cr_bob, name=f"Tele_{basis}")

    # State preparation on qubit 0
    input_state = np.array(input_state, dtype=complex)
    norm = np.linalg.norm(input_state)
    if abs(norm - 1.0) > 1e-6:
        input_state = input_state / norm
    _apply_state_prep_gates(qc, 0, input_state)

    # Bell pair (qubits 1, 2)
    qc.h(1)
    qc.cx(1, 2)

    # Alice's BSM
    qc.cx(0, 1)
    qc.h(0)

    # Measure Alice's qubits
    qc.measure(0, cr_alice[0])
    qc.measure(1, cr_alice[1])

    # Bob's conditional corrections (using dynamic circuits / if_test)
    with qc.if_test((cr_alice[1], 1)):
        qc.x(2)
    with qc.if_test((cr_alice[0], 1)):
        qc.z(2)

    # Tomography rotation on Bob's qubit before final measurement
    if basis == 'X':
        qc.h(2)          # Rotate to X basis
    elif basis == 'Y':
        qc.sdg(2)        # S† then H rotates to Y basis
        qc.h(2)
    # 'Z' needs no rotation

    # Measure Bob's qubit
    qc.measure(2, cr_bob[0])

    return qc


def reconstruct_density_matrix(tomo_counts, shots_per_basis):
    """
    Reconstruct a single-qubit density matrix from tomography counts.

    Parameters
    ----------
    tomo_counts : dict
        {basis: {bitstring: count}} for bases 'X', 'Y', 'Z'.
        Bitstrings are for Bob's qubit only (after marginalizing Alice).
    shots_per_basis : int
        Number of shots per measurement basis.

    Returns
    -------
    np.ndarray (2, 2)
        Reconstructed density matrix.
    """
    # Compute expectation values <sigma_i>
    expectations = {}
    for basis in ['X', 'Y', 'Z']:
        counts = tomo_counts[basis]
        n_0 = counts.get(0, 0)  # Bob measured |0>
        n_1 = counts.get(1, 0)  # Bob measured |1>
        total = n_0 + n_1
        if total == 0:
            expectations[basis] = 0.0
        else:
            expectations[basis] = (n_0 - n_1) / total

    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # rho = (I + <X>X + <Y>Y + <Z>Z) / 2
    rho = (I + expectations['X'] * X +
           expectations['Y'] * Y +
           expectations['Z'] * Z) / 2

    # Ensure physical (positive semidefinite) via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)
    eigvals /= np.sum(eigvals)
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

    return rho


def _marginalize_bob_counts(raw_counts):
    """
    Extract Bob's qubit measurement from the combined bitstring.

    The circuit has registers: alice (2 bits), bob (1 bit).
    Qiskit returns bitstrings in reverse register order: 'bob alice1 alice0'.
    So the leftmost bit is Bob's measurement.

    Parameters
    ----------
    raw_counts : dict
        {bitstring: count} from hardware execution.

    Returns
    -------
    dict
        {0: count_0, 1: count_1} for Bob's qubit only.
    """
    bob_counts = {0: 0, 1: 0}
    for bitstring, count in raw_counts.items():
        # bitstring format: 'bob alice1 alice0' (space-separated registers)
        # or combined like '0 01' or just '001'
        # Need to extract the bob bit
        bits = bitstring.replace(" ", "")
        bob_bit = int(bits[0])  # Leftmost bit is last register (bob)
        bob_counts[bob_bit] += count
    return bob_counts


def run_hardware_validation(
    backend_name="ibm_torino",
    shots=8192,
    optimization_level=3,
    verbose=True,
):
    """
    Execute teleportation on IBM hardware and compare with simulation.

    Parameters
    ----------
    backend_name : str
        IBM backend name.
    shots : int
        Number of shots per circuit.
    optimization_level : int
        Transpiler optimization level (0-3).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Hardware validation results with per-state fidelities and comparison.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    if verbose:
        print(f"\n  Connecting to IBM Quantum...")

    service = QiskitRuntimeService(channel='ibm_quantum_platform')
    backend = service.backend(backend_name)

    if verbose:
        print(f"  Backend: {backend.name} ({backend.num_qubits} qubits)")
        status = backend.status()
        print(f"  Status: {status.status_msg}, pending: {status.pending_jobs}")

    # Test states (same as simulation)
    test_states = [
        {"name": "|0>",  "vector": np.array([1, 0], dtype=complex)},
        {"name": "|1>",  "vector": np.array([0, 1], dtype=complex)},
        {"name": "|+>",  "vector": np.array([1, 1], dtype=complex) / np.sqrt(2)},
        {"name": "|->",  "vector": np.array([1, -1], dtype=complex) / np.sqrt(2)},
        {"name": "|+i>", "vector": np.array([1, 1j], dtype=complex) / np.sqrt(2)},
        {"name": "|-i>", "vector": np.array([1, -1j], dtype=complex) / np.sqrt(2)},
    ]

    bases = ['X', 'Y', 'Z']

    # Build all circuits (6 states × 3 bases = 18 circuits)
    circuits = []
    circuit_map = []  # Track (state_idx, basis) for each circuit
    for si, state in enumerate(test_states):
        for basis in bases:
            qc = build_teleportation_tomography_circuits(state["vector"], basis)
            circuits.append(qc)
            circuit_map.append((si, basis))

    if verbose:
        print(f"  Built {len(circuits)} tomography circuits (6 states × 3 bases)")

    # Transpile for hardware
    if verbose:
        print(f"  Transpiling (optimization_level={optimization_level})...")

    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
    )
    transpiled = pm.run(circuits)

    if verbose:
        depths = [qc.depth() for qc in transpiled]
        print(f"  Transpiled depths: min={min(depths)}, max={max(depths)}, "
              f"mean={np.mean(depths):.0f}")

    # Execute on hardware
    if verbose:
        print(f"  Submitting {len(transpiled)} circuits ({shots} shots each)...")
        t_submit = time.time()

    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=shots)

    if verbose:
        print(f"  Job ID: {job.job_id()}")
        print(f"  Waiting for results...")

    result = job.result()

    if verbose:
        t_elapsed = time.time() - t_submit
        print(f"  Results received ({t_elapsed:.0f}s)")

    # Process results
    hw_results = []
    for si, state in enumerate(test_states):
        tomo_counts = {}
        for basis in bases:
            # Find the circuit index for this (state, basis)
            ci = si * len(bases) + bases.index(basis)
            pub_result = result[ci]

            # Extract counts - SamplerV2 returns BitArray
            # Get counts from the classical registers
            data = pub_result.data

            # Marginalize: we need Bob's qubit counts
            # The registers are 'alice' (2 bits) and 'bob' (1 bit)
            try:
                # Try to get bob register directly
                bob_bits = data.bob
                bob_array = bob_bits.get_int_counts()
                tomo_counts[basis] = {0: bob_array.get(0, 0), 1: bob_array.get(1, 0)}
            except AttributeError:
                # Fallback: get combined counts and marginalize
                try:
                    combined = data.meas
                    raw_counts = combined.get_counts()
                    tomo_counts[basis] = _marginalize_bob_counts(raw_counts)
                except AttributeError:
                    # Another fallback: iterate over available creg names
                    for attr_name in dir(data):
                        if not attr_name.startswith('_'):
                            try:
                                cr_data = getattr(data, attr_name)
                                if hasattr(cr_data, 'get_int_counts'):
                                    raw_counts = cr_data.get_int_counts()
                                    if verbose and si == 0 and basis == 'Z':
                                        print(f"  Found register: {attr_name}, "
                                              f"counts: {raw_counts}")
                            except Exception:
                                pass
                    # Use joint bitstring
                    all_cregs = [attr for attr in dir(data)
                                 if not attr.startswith('_') and
                                 hasattr(getattr(data, attr, None), 'get_int_counts')]
                    if all_cregs:
                        # Get the last register (bob) if multiple exist
                        bob_reg = all_cregs[-1]
                        bob_ints = getattr(data, bob_reg).get_int_counts()
                        tomo_counts[basis] = {0: bob_ints.get(0, 0),
                                              1: bob_ints.get(1, 0)}
                    else:
                        raise RuntimeError(f"Cannot extract measurement data from result {ci}")

        # Reconstruct density matrix
        rho_hw = reconstruct_density_matrix(tomo_counts, shots)

        # Compute fidelity against ideal state
        ideal_dm = np.outer(state["vector"], state["vector"].conj())
        fid_hw = float(np.real(np.trace(ideal_dm @ rho_hw)))
        # Ensure physical range
        fid_hw = max(0.0, min(1.0, fid_hw))

        purity = float(np.real(np.trace(rho_hw @ rho_hw)))

        hw_results.append({
            "input_name": state["name"],
            "fidelity": fid_hw,
            "purity": purity,
            "density_matrix": rho_hw,
            "tomo_counts": tomo_counts,
        })

        if verbose:
            print(f"  {state['name']:>6s}  →  F = {fid_hw:.4f}  "
                  f"Purity = {purity:.4f}")

    # Summary
    hw_fids = [r["fidelity"] for r in hw_results]
    mean_fid = np.mean(hw_fids)
    std_fid = np.std(hw_fids)

    if verbose:
        print(f"\n  Hardware mean fidelity: {mean_fid:.4f} ± {std_fid:.4f}")
        print(f"  Above classical (2/3): {'YES' if mean_fid > 2/3 else 'NO'}")

    return {
        "backend": backend_name,
        "shots": shots,
        "num_qubits_backend": backend.num_qubits,
        "results": hw_results,
        "mean_fidelity": float(mean_fid),
        "std_fidelity": float(std_fid),
        "job_id": job.job_id(),
    }


def compare_hardware_simulation(hw_results, sim_results, verbose=True):
    """
    Compare hardware and simulation fidelities.

    Parameters
    ----------
    hw_results : dict
        Output of run_hardware_validation().
    sim_results : list[dict]
        Output of simulate_all_test_states().
    verbose : bool
        Print comparison table.

    Returns
    -------
    dict
        Comparison metrics.
    """
    # Build lookup
    sim_lookup = {r["input_name"]: r["fidelity"] for r in sim_results}

    comparisons = []
    for hr in hw_results["results"]:
        name = hr["input_name"]
        # Map hardware names to simulation names
        name_map = {
            "|0>": "|0\u27e9", "|1>": "|1\u27e9",
            "|+>": "|+\u27e9", "|->": "|-\u27e9",
            "|+i>": "|+i\u27e9", "|-i>": "|-i\u27e9",
        }
        sim_name = name_map.get(name, name)
        sim_fid = sim_lookup.get(sim_name, None)
        if sim_fid is None:
            continue

        gap = hr["fidelity"] - sim_fid
        comparisons.append({
            "state": name,
            "hw_fidelity": hr["fidelity"],
            "sim_fidelity": sim_fid,
            "gap": gap,
            "abs_gap": abs(gap),
            "within_5pct": abs(gap) < 0.05,
        })

    mean_gap = np.mean([c["abs_gap"] for c in comparisons])
    max_gap = max(c["abs_gap"] for c in comparisons)
    all_within = all(c["within_5pct"] for c in comparisons)

    if verbose:
        print(f"\n  {'State':>6s}  {'HW F':>7s}  {'Sim F':>7s}  {'Gap':>7s}  Status")
        print(f"  {'-'*45}")
        for c in comparisons:
            status = "OK" if c["within_5pct"] else "REVIEW"
            sign = "+" if c["gap"] >= 0 else ""
            print(f"  {c['state']:>6s}  {c['hw_fidelity']:>7.4f}  "
                  f"{c['sim_fidelity']:>7.4f}  {sign}{c['gap']:>6.4f}  {status}")

        print(f"\n  Mean |gap|: {mean_gap:.4f}")
        print(f"  Max  |gap|: {max_gap:.4f}")
        print(f"  All within 5%: {'YES' if all_within else 'NO'}")

        if mean_gap < 0.05:
            print(f"  VALIDATION: PASSED — simulation matches hardware")
        elif mean_gap < 0.10:
            print(f"  VALIDATION: PARTIAL — reasonable agreement, noise model could be tuned")
        else:
            print(f"  VALIDATION: NEEDS WORK — significant gap, noise model needs revision")

    return {
        "comparisons": comparisons,
        "mean_gap": float(mean_gap),
        "max_gap": float(max_gap),
        "all_within_5pct": all_within,
        "hw_mean": float(hw_results["mean_fidelity"]),
        "sim_mean": float(np.mean([c["sim_fidelity"] for c in comparisons])),
    }
