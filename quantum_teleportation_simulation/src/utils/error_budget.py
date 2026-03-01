"""
Error budget analysis.

Computes and formats the contribution of each noise source
to total teleportation infidelity.
"""

import numpy as np


def compute_cumulative_fidelity(error_budget):
    """
    Compute cumulative fidelity decay through protocol phases.

    Assumes errors are multiplicative (independent).

    Parameters
    ----------
    error_budget : dict
        From ComprehensiveTeleportationNoise.compute_error_budget().

    Returns
    -------
    list[dict]
        Ordered list of (source, infidelity, cumulative_fidelity).
    """
    # Order of error application in protocol
    ordered_sources = [
        "state_preparation",
        "single_qubit_errors",  # H gate
        "CNOT_gate_errors",     # Bell prep + Alice CNOT
        "T1_decay",
        "T2_dephasing",
        "readout_errors",
        "leakage",
        "crosstalk",
        "one_over_f_noise",
        "coherent_errors",
    ]

    cumulative = 1.0
    trajectory = []

    for source in ordered_sources:
        if source in error_budget and source != "TOTAL_infidelity_estimate":
            infidelity = error_budget[source]["absolute"]
            cumulative *= (1 - infidelity)
            trajectory.append({
                "source": source,
                "infidelity_contribution": infidelity,
                "cumulative_fidelity": cumulative,
            })

    return trajectory


def format_error_budget(error_budget):
    """
    Format error budget as a printable table.

    Parameters
    ----------
    error_budget : dict
        From ComprehensiveTeleportationNoise.compute_error_budget().

    Returns
    -------
    str
        Formatted table string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ERROR BUDGET ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"{'Source':<30} {'Contribution':>12} {'Phase':<25}")
    lines.append("-" * 70)

    total_key = "TOTAL_infidelity_estimate"
    for source, data in error_budget.items():
        if source == total_key:
            continue
        pct = data["percent"]
        phase = data.get("phase", "")
        lines.append(f"  {source:<28} {pct:>10.3f}%   {phase:<25}")

    lines.append("-" * 70)
    if total_key in error_budget:
        total = error_budget[total_key]
        lines.append(f"  {'TOTAL (first-order sum)':<28} {total['percent']:>10.3f}%")
        expected_fid = 1 - total["absolute"]
        lines.append(f"  {'Expected fidelity':<28} {expected_fid:>10.4f}")

        # Multiplicative estimate (more accurate)
        mult_fid = 1.0
        for source, data in error_budget.items():
            if source != total_key:
                mult_fid *= (1 - data["absolute"])
        lines.append(f"  {'Multiplicative estimate':<28} {mult_fid:>10.4f}")

    lines.append("=" * 70)
    return "\n".join(lines)


def dominant_error_sources(error_budget, top_n=5):
    """
    Return the top N dominant error sources.

    Parameters
    ----------
    error_budget : dict
    top_n : int

    Returns
    -------
    list[tuple]
        [(source_name, percent_contribution), ...]
    """
    items = [
        (k, v["percent"])
        for k, v in error_budget.items()
        if k != "TOTAL_infidelity_estimate"
    ]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_n]
