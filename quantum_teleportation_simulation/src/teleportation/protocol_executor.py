"""
Teleportation protocol executor.

Runs the complete teleportation protocol with MEASUREMENT-BASED simulation:
- Density matrix simulation up to Alice's BSM
- Analytic projective measurement with readout errors
- Idle decoherence on Bob's qubit during measurement + classical delay
- Noisy conditional corrections by Bob
- Weighted average over all 4×4 = 16 (true outcome × received outcome) branches
"""

import numpy as np
import time
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, state_fidelity, Statevector, partial_trace

from .teleportation_circuit import (
    create_pre_measurement_circuit,
    create_teleportation_circuit_deferred,
    get_test_states,
)
from .bell_preparation import simulate_bell_preparation


# ── Pauli matrices ──────────────────────────────────────────────────────
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _apply_thermal_relaxation_2x2(rho, T1, T2, duration):
    """
    Apply thermal relaxation channel to a 2×2 density matrix.

    Models amplitude damping (T1) and pure dephasing (T2 - T1/2).

    Parameters
    ----------
    rho : np.ndarray (2,2)
        Input density matrix.
    T1, T2, duration : float
        Coherence times and idle duration (all in seconds).

    Returns
    -------
    np.ndarray (2,2)
    """
    if T1 <= 0 or duration <= 0:
        return rho.copy()
    T2 = min(T2, 2 * T1)
    gamma = 1 - np.exp(-duration / T1)
    # Pure dephasing factor
    if T2 > 0 and T1 > 0:
        rate_phi = 1.0 / T2 - 1.0 / (2 * T1)
        lambda_phi = np.exp(-duration * max(rate_phi, 0))
    else:
        lambda_phi = 0.0

    rho_out = np.zeros((2, 2), dtype=complex)
    rho_out[0, 0] = rho[0, 0] + gamma * rho[1, 1]
    rho_out[1, 1] = (1 - gamma) * rho[1, 1]
    rho_out[0, 1] = np.sqrt(1 - gamma) * lambda_phi * rho[0, 1]
    rho_out[1, 0] = np.conj(rho_out[0, 1])
    return rho_out


def _apply_gate_with_noise(rho, gate_matrix, gate_error, T1, T2, gate_duration):
    """
    Apply a unitary gate followed by depolarizing + thermal relaxation.

    Parameters
    ----------
    rho : np.ndarray (2,2)
    gate_matrix : np.ndarray (2,2)
    gate_error : float
        Depolarizing error probability.
    T1, T2, gate_duration : float

    Returns
    -------
    np.ndarray (2,2)
    """
    # Ideal gate
    rho_g = gate_matrix @ rho @ gate_matrix.conj().T
    # Depolarizing noise
    rho_g = (1 - gate_error) * rho_g + gate_error * _I / 2
    # Thermal relaxation during gate
    rho_g = _apply_thermal_relaxation_2x2(rho_g, T1, T2, gate_duration)
    return rho_g


class TeleportationProtocolSimulator:
    """
    Complete teleportation protocol simulator with comprehensive noise.

    Supports:
    - Density matrix simulation (exact fidelity)
    - Multiple input states
    - Error budget tracking
    - Phase-by-phase analysis
    """

    def __init__(self, noise_model_obj, params):
        """
        Parameters
        ----------
        noise_model_obj : ComprehensiveTeleportationNoise
            The comprehensive noise model.
        params : dict
            Device parameters.
        """
        self.noise_obj = noise_model_obj
        self.params = params
        self.results = []

    def simulate_single_state(self, input_state, input_name="unknown"):
        """
        Simulate teleportation with measurement-based corrections.

        This properly models:
        1. Noisy pre-measurement circuit (state prep, Bell pair, Alice's BSM)
        2. Projective measurement of Alice's qubits (0, 1)
        3. Readout errors causing Bob to receive wrong outcome
        4. Idle decoherence on Bob's qubit during measurement + classical delay
        5. Noisy correction gates (X and/or Z) applied by Bob based on received bits

        The final fidelity is the weighted average over all 16 branches:
        (4 true outcomes) × (4 received outcomes due to readout errors).

        Parameters
        ----------
        input_state : np.ndarray
            2-element state vector [alpha, beta].
        input_name : str
            Human-readable name for the input state.

        Returns
        -------
        dict
            Simulation results including fidelity, density matrix, etc.
        """
        p = self.params
        t_start = time.time()

        # ── Step 1: Run pre-measurement circuit with noise ──────────────
        qc = create_pre_measurement_circuit(input_state)
        qc.save_density_matrix()

        noise_model = self.noise_obj.build_noise_model(qubits_used=[0, 1, 2])
        sim = AerSimulator(method="density_matrix", noise_model=noise_model)
        result = sim.run(qc, shots=1).result()
        rho_full = result.data()["density_matrix"]  # 8×8 numpy array

        # ── Step 2: Readout error probabilities for qubits 0 and 1 ──────
        # Only apply readout errors if SPAM is enabled in the noise model
        if self.noise_obj.flags.get("spam", True):
            e0 = p["readout_error_per_qubit"][0] * self.noise_obj.noise_scale
            e1 = p["readout_error_per_qubit"][1] * self.noise_obj.noise_scale
            e0 = min(e0, 0.5)
            e1 = min(e1, 0.5)
        else:
            e0 = 0.0
            e1 = 0.0

        # ── Step 3: Bob's qubit parameters (qubit 2) ────────────────────
        # Only apply idle / correction noise if thermal is enabled
        apply_idle = self.noise_obj.flags.get("thermal", True)
        apply_gate_err = self.noise_obj.flags.get("depolarizing", True)

        T1_bob = p["T1_per_qubit"][2] if 2 < len(p["T1_per_qubit"]) else p["T1"]
        T2_bob = p["T2_per_qubit"][2] if 2 < len(p["T2_per_qubit"]) else p["T2"]
        T2_bob = min(T2_bob, 2 * T1_bob)
        # Scale coherence times by noise scale (for ZNE)
        T1_bob_eff = T1_bob / self.noise_obj.noise_scale
        T2_bob_eff = T2_bob / self.noise_obj.noise_scale
        T2_bob_eff = min(T2_bob_eff, 2 * T1_bob_eff)

        # Idle time: measurement + classical communication
        t_idle = (p["measurement_duration_ns"] + p["classical_delay_ns"]) * 1e-9
        t_gate = p["gate_duration_single_ns"] * 1e-9
        sq_error = p["single_qubit_error_per_qubit"][2] if 2 < len(
            p["single_qubit_error_per_qubit"]) else p["single_qubit_error"]
        sq_error_scaled = min(sq_error * self.noise_obj.noise_scale, 1.0) if apply_gate_err else 0.0

        # ── Step 4: Iterate over all measurement branches ───────────────
        # Qiskit uses little-endian: basis index = q2*4 + q1*2 + q0
        rho_bob_final = np.zeros((2, 2), dtype=complex)

        for m0 in range(2):
            for m1 in range(2):
                # Extract Bob's reduced state for true outcome (m0, m1)
                idx_base = m1 * 2 + m0  # offset for q0=m0, q1=m1
                # Bob's 2×2 block: rows/cols with q2 = 0 or 1
                rho_bob_branch = np.zeros((2, 2), dtype=complex)
                for a in range(2):
                    for b in range(2):
                        rho_bob_branch[a, b] = rho_full[a * 4 + idx_base,
                                                        b * 4 + idx_base]

                # Probability of this true outcome
                p_outcome = np.real(np.trace(rho_bob_branch))
                if p_outcome < 1e-15:
                    continue

                # Normalize Bob's state
                rho_bob_branch /= p_outcome

                # Apply idle decoherence during measurement + classical delay
                if apply_idle:
                    rho_bob_branch = _apply_thermal_relaxation_2x2(
                        rho_bob_branch, T1_bob_eff, T2_bob_eff, t_idle
                    )

                # For each received outcome (b0, b1) due to readout errors
                for b0 in range(2):
                    for b1 in range(2):
                        # Readout confusion probability
                        p_b0 = (1 - e0) if b0 == m0 else e0
                        p_b1 = (1 - e1) if b1 == m1 else e1
                        p_readout = p_b0 * p_b1

                        rho_branch = rho_bob_branch.copy()

                        # Bob applies corrections based on received (b0, b1)
                        # b1 = 1 → apply X correction
                        if b1 == 1:
                            rho_branch = _apply_gate_with_noise(
                                rho_branch, _X, sq_error_scaled,
                                T1_bob_eff, T2_bob_eff,
                                t_gate if apply_idle else 0.0
                            )
                        elif apply_idle:
                            # Even if no X applied, thermal relaxation during
                            # the time Bob would check/decide
                            rho_branch = _apply_thermal_relaxation_2x2(
                                rho_branch, T1_bob_eff, T2_bob_eff, t_gate
                            )

                        # b0 = 1 → apply Z correction
                        if b0 == 1:
                            rho_branch = _apply_gate_with_noise(
                                rho_branch, _Z, sq_error_scaled,
                                T1_bob_eff, T2_bob_eff,
                                t_gate if apply_idle else 0.0
                            )
                        elif apply_idle:
                            rho_branch = _apply_thermal_relaxation_2x2(
                                rho_branch, T1_bob_eff, T2_bob_eff, t_gate
                            )

                        # Weight this branch
                        rho_bob_final += p_outcome * p_readout * rho_branch

        # ── Step 5: Compute fidelity ────────────────────────────────────
        rho_ideal = np.outer(input_state, input_state.conj())  # |ψ⟩⟨ψ|
        # Fidelity: F = Tr(ρ_ideal · ρ_bob)  (for pure ideal state)
        fid = np.real(np.trace(rho_ideal @ rho_bob_final))
        fid = float(np.clip(fid, 0.0, 1.0))

        # Purity
        purity = float(np.real(np.trace(rho_bob_final @ rho_bob_final)))

        t_elapsed = time.time() - t_start

        result_dict = {
            "input_name": input_name,
            "input_state": input_state.tolist(),
            "fidelity": fid,
            "purity": purity,
            "density_matrix_bob": DensityMatrix(rho_bob_final),
            "simulation_time_s": t_elapsed,
        }

        self.results.append(result_dict)
        return result_dict

    def simulate_all_test_states(self, verbose=True):
        """
        Simulate teleportation for all 6 standard test states.

        Parameters
        ----------
        verbose : bool
            If True, print progress. If False, run silently.

        Returns
        -------
        list[dict]
            Results for each test state.
        """
        test_states = get_test_states()
        all_results = []

        if verbose:
            print("=" * 65)
            print("QUANTUM TELEPORTATION SIMULATION - ALL TEST STATES")
            print("=" * 65)
            print(f"Noise sources: {self.noise_obj.get_enabled_sources()}")
            print(f"Noise scale: {self.noise_obj.noise_scale:.2f}x")
            print("-" * 65)

        for ts in test_states:
            res = self.simulate_single_state(ts["vector"], ts["name"])
            all_results.append(res)
            if verbose:
                print(
                    f"  {ts['name']:6s}  →  F = {res['fidelity']:.4f}  "
                    f"Purity = {res['purity']:.4f}  "
                    f"({res['simulation_time_s']:.2f}s)"
                )

        # Summary statistics
        fidelities = [r["fidelity"] for r in all_results]
        mean_fid = np.mean(fidelities)
        std_fid = np.std(fidelities)

        if verbose:
            print("-" * 65)
            print(f"  Mean fidelity:  {mean_fid:.4f} ± {std_fid:.4f}")
            print(f"  Min fidelity:   {min(fidelities):.4f}")
            print(f"  Max fidelity:   {max(fidelities):.4f}")
            print(f"  Above classical (2/3): {'YES' if mean_fid > 2/3 else 'NO'}")
            print("=" * 65)

        return all_results

    def simulate_with_shots(self, input_state, input_name="unknown",
                             n_trials=20):
        """
        Run multiple teleportation simulations with parameter perturbation
        to estimate fidelity distribution (calibration drift model).

        Parameters
        ----------
        input_state : np.ndarray
            Input state vector.
        input_name : str
            Name of the input state.
        n_trials : int
            Number of independent trials with perturbed parameters.

        Returns
        -------
        dict
            Statistics: mean, std, confidence interval.
        """
        from ..noise_models.composite_noise import ComprehensiveTeleportationNoise

        rng = np.random.default_rng()
        fidelities = []

        for trial in range(n_trials):
            # Perturb parameters (calibration drift)
            mc_params = self.params.copy()
            drift = 1.0 + 0.05 * rng.standard_normal()
            drift = max(drift, 0.5)
            mc_params["T1"] = self.params["T1"] * drift
            mc_params["T2"] = min(self.params["T2"] * drift, 2 * mc_params["T1"])
            mc_params["T1_per_qubit"] = [t * drift for t in self.params["T1_per_qubit"]]
            mc_params["T2_per_qubit"] = [min(t * drift, 2 * mc_params["T1"])
                                          for t in self.params["T2_per_qubit"]]
            drift_g = 1.0 + 0.10 * rng.standard_normal()
            drift_g = max(drift_g, 0.1)
            mc_params["cnot_error_mean"] = self.params["cnot_error_mean"] * drift_g
            mc_params["cnot_errors"] = {k: v * drift_g
                                         for k, v in self.params["cnot_errors"].items()}

            mc_noise = ComprehensiveTeleportationNoise(mc_params,
                                                       noise_scale=self.noise_obj.noise_scale)
            mc_sim = TeleportationProtocolSimulator(mc_noise, mc_params)
            res = mc_sim.simulate_single_state(input_state, input_name)
            fidelities.append(res["fidelity"])

        mean_fid = np.mean(fidelities)
        std_fid = np.std(fidelities)
        stderr = std_fid / np.sqrt(n_trials)
        ci_95 = 1.96 * stderr

        return {
            "input_name": input_name,
            "mean_fidelity": mean_fid,
            "std": std_fid,
            "stderr": stderr,
            "ci_95_lower": mean_fid - ci_95,
            "ci_95_upper": mean_fid + ci_95,
            "n_trials": n_trials,
            "all_fidelities": fidelities,
        }

    def compare_noise_models(self, input_state, input_name="|+⟩"):
        """
        Compare teleportation fidelity across different noise configurations:
        1. Ideal (no noise)
        2. Markovian only (T1/T2)
        3. Depolarizing only
        4. Comprehensive (all sources)

        Returns
        -------
        dict
            Fidelity for each noise configuration.
        """
        from ..noise_models.composite_noise import (
            ComprehensiveTeleportationNoise,
            build_markovian_only_noise,
            build_depolarizing_only_noise,
        )

        configs = {}

        # 1. Ideal (no noise)
        ideal_model = ComprehensiveTeleportationNoise(
            self.params,
            enable_thermal=False, enable_depolarizing=False,
            enable_crosstalk=False, enable_leakage=False,
            enable_1f=False, enable_spam=False,
            enable_coherent=False, enable_collective=False,
        )
        sim_ideal = TeleportationProtocolSimulator(ideal_model, self.params)
        res_ideal = sim_ideal.simulate_single_state(input_state, input_name)
        configs["ideal"] = res_ideal["fidelity"]

        # 2. Markovian only
        markov_model = build_markovian_only_noise(self.params)
        sim_markov = TeleportationProtocolSimulator(markov_model, self.params)
        res_markov = sim_markov.simulate_single_state(input_state, input_name)
        configs["markovian_only"] = res_markov["fidelity"]

        # 3. Depolarizing only
        depol_model = build_depolarizing_only_noise(self.params)
        sim_depol = TeleportationProtocolSimulator(depol_model, self.params)
        res_depol = sim_depol.simulate_single_state(input_state, input_name)
        configs["depolarizing_only"] = res_depol["fidelity"]

        # 4. Comprehensive (current model)
        res_comp = self.simulate_single_state(input_state, input_name)
        configs["comprehensive"] = res_comp["fidelity"]

        # Remove from accumulated results
        self.results = self.results[:-1]

        return configs

    def protocol_timeline(self):
        """
        Generate the protocol timeline with timing and noise sources.

        Returns
        -------
        list[dict]
            Timeline entries.
        """
        p = self.params
        t = 0

        phases = []

        # Phase 1: Bell pair preparation
        dur = p["gate_duration_single_ns"] + p["gate_duration_cnot_ns"]
        phases.append({
            "phase_number": 1,
            "phase_name": "Bell pair preparation",
            "start_time_ns": t,
            "end_time_ns": t + dur,
            "duration_ns": dur,
            "active_qubits": [1, 2],
            "idle_qubits": [0],
            "noise_sources": ["gate_errors", "T1/T2", "leakage"],
        })
        t += dur

        # Phase 2: Alice CNOT
        dur = p["gate_duration_cnot_ns"]
        phases.append({
            "phase_number": 2,
            "phase_name": "Alice CNOT(q0,q1)",
            "start_time_ns": t,
            "end_time_ns": t + dur,
            "duration_ns": dur,
            "active_qubits": [0, 1],
            "idle_qubits": [2],
            "noise_sources": ["gate_errors", "T1/T2", "crosstalk", "leakage"],
        })
        t += dur

        # Phase 3: Alice H
        dur = p["gate_duration_single_ns"]
        phases.append({
            "phase_number": 3,
            "phase_name": "Alice H(q0)",
            "start_time_ns": t,
            "end_time_ns": t + dur,
            "duration_ns": dur,
            "active_qubits": [0],
            "idle_qubits": [1, 2],
            "noise_sources": ["gate_errors", "T1/T2"],
        })
        t += dur

        # Phase 4: Measurement
        dur = p["measurement_duration_ns"]
        phases.append({
            "phase_number": 4,
            "phase_name": "Bell measurement",
            "start_time_ns": t,
            "end_time_ns": t + dur,
            "duration_ns": dur,
            "active_qubits": [0, 1],
            "idle_qubits": [2],
            "noise_sources": ["readout_errors", "T1/T2_idle"],
        })
        t += dur

        # Phase 5: Classical delay
        dur = p["classical_delay_ns"]
        phases.append({
            "phase_number": 5,
            "phase_name": "Classical communication",
            "start_time_ns": t,
            "end_time_ns": t + dur,
            "duration_ns": dur,
            "active_qubits": [],
            "idle_qubits": [2],
            "noise_sources": ["T1/T2_idle"],
        })
        t += dur

        # Phase 6: Bob's corrections
        dur = 2 * p["gate_duration_single_ns"]
        phases.append({
            "phase_number": 6,
            "phase_name": "Bob's corrections",
            "start_time_ns": t,
            "end_time_ns": t + dur,
            "duration_ns": dur,
            "active_qubits": [2],
            "idle_qubits": [],
            "noise_sources": ["gate_errors", "T1/T2"],
        })
        t += dur

        return phases, t
