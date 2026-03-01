"""
Device parameters loader and manager.

Loads experimentally-justified device parameters from JSON and provides
them in a convenient format for noise model construction.
"""

import json
import os
import numpy as np


def get_default_params_path():
    """Return the default parameter file path."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "parameters", "justified_parameters.json")


def load_device_parameters(device_name="ibmq_manila", params_path=None):
    """
    Load device parameters from the justified parameters JSON.

    Parameters
    ----------
    device_name : str
        Name of the device (key in JSON).
    params_path : str, optional
        Path to the JSON file. Uses default if None.

    Returns
    -------
    dict
        Flat dictionary of device parameters ready for simulation use.
    """
    if params_path is None:
        params_path = get_default_params_path()

    with open(params_path, "r") as f:
        all_params = json.load(f)

    raw = all_params[device_name]

    # Build a flat, convenient parameter dict
    params = {
        "device_name": device_name,
        "device_type": raw["device_type"],
        "num_qubits": raw["num_qubits"],
        "connectivity": raw["connectivity"],
        "coupling_map": [tuple(pair) for pair in raw["coupling_map"]],

        # Coherence times (seconds)
        "T1": raw["T1"]["mean"],
        "T1_per_qubit": raw["T1"]["values_per_qubit"],
        "T2": raw["T2"]["mean"],
        "T2_per_qubit": raw["T2"]["values_per_qubit"],

        # Gate errors (probabilities)
        "single_qubit_error": raw["single_qubit_gate_error"]["mean"],
        "single_qubit_error_per_qubit": raw["single_qubit_gate_error"]["values_per_qubit"],
        "cnot_error_mean": raw["cnot_gate_error"]["mean"],
        "cnot_errors": {
            k: v for k, v in raw["cnot_gate_error"].items()
            if k not in ("mean", "source")
        },

        # Readout
        "readout_error_per_qubit": raw["readout_error"]["per_qubit"],
        "readout_error_mean": raw["readout_error"]["mean"],

        # Gate durations (seconds)
        "gate_duration_h": raw["gate_durations"]["single_qubit_ns"] * 1e-9,
        "gate_duration_x": raw["gate_durations"]["single_qubit_ns"] * 1e-9,
        "gate_duration_z": raw["gate_durations"]["single_qubit_ns"] * 1e-9,
        "gate_duration_cnot": raw["gate_durations"]["cnot_ns"] * 1e-9,
        "measurement_duration": raw["gate_durations"]["measurement_ns"] * 1e-9,
        "gate_duration_single_ns": raw["gate_durations"]["single_qubit_ns"],
        "gate_duration_cnot_ns": raw["gate_durations"]["cnot_ns"],
        "measurement_duration_ns": raw["gate_durations"]["measurement_ns"],

        # ZZ coupling (Hz)
        "zz_coupling": {
            k: v for k, v in raw["zz_coupling_Hz"].items()
            if k != "source"
        },

        # Leakage
        "leakage_single": raw["leakage_rates"]["single_qubit"],
        "leakage_cnot": raw["leakage_rates"]["cnot"],

        # 1/f noise
        "one_over_f_alpha": raw["one_over_f_noise"]["alpha"],
        "one_over_f_amplitude": raw["one_over_f_noise"]["amplitude"],

        # Thermal / SPAM
        "thermal_population": raw["thermal_population"]["value"],

        # Classical communication
        "classical_delay": raw["classical_communication_delay_ns"] * 1e-9,
        "classical_delay_ns": raw["classical_communication_delay_ns"],

        # Coherent errors
        "systematic_over_rotation": raw["systematic_over_rotation"],
    }

    return params


def get_qubit_T1(params, qubit):
    """Get T1 for a specific qubit."""
    if qubit < len(params["T1_per_qubit"]):
        return params["T1_per_qubit"][qubit]
    return params["T1"]


def get_qubit_T2(params, qubit):
    """Get T2 for a specific qubit."""
    if qubit < len(params["T2_per_qubit"]):
        return params["T2_per_qubit"][qubit]
    return params["T2"]


def get_cnot_error(params, control, target):
    """Get CNOT error for a specific qubit pair."""
    key = f"({control},{target})"
    return params["cnot_errors"].get(key, params["cnot_error_mean"])


def get_readout_error(params, qubit):
    """Get readout error for a specific qubit."""
    if qubit < len(params["readout_error_per_qubit"]):
        return params["readout_error_per_qubit"][qubit]
    return params["readout_error_mean"]
