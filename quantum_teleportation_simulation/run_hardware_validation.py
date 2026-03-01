#!/usr/bin/env python3
"""
Run hardware validation for quantum teleportation.

Submits teleportation circuits to real IBM Quantum hardware,
performs state tomography, and compares with simulation results.

Usage:
    python run_hardware_validation.py [--backend ibm_torino] [--shots 8192]
"""

import sys
import os
import json
import csv
import time
import argparse
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.validation.hardware_validation import (
    run_hardware_validation,
    compare_hardware_simulation,
    build_teleportation_tomography_circuits,
)


def save_results_csv(hw_results, comparison, output_dir):
    """Save hardware results and comparison to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Hardware results
    csv_path = os.path.join(output_dir, "hardware_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["State", "HW_Fidelity", "Purity",
                         "Z_counts_0", "Z_counts_1",
                         "X_counts_0", "X_counts_1",
                         "Y_counts_0", "Y_counts_1"])
        for r in hw_results["results"]:
            tc = r["tomo_counts"]
            writer.writerow([
                r["input_name"],
                f"{r['fidelity']:.6f}",
                f"{r['purity']:.6f}",
                tc['Z'].get(0, 0), tc['Z'].get(1, 0),
                tc['X'].get(0, 0), tc['X'].get(1, 0),
                tc['Y'].get(0, 0), tc['Y'].get(1, 0),
            ])
    print(f"  Saved: {csv_path}")

    # Comparison
    if comparison:
        csv_path2 = os.path.join(output_dir, "hardware_vs_simulation.csv")
        with open(csv_path2, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["State", "HW_Fidelity", "Sim_Fidelity",
                             "Gap", "Within_5pct"])
            for c in comparison["comparisons"]:
                writer.writerow([
                    c["state"],
                    f"{c['hw_fidelity']:.6f}",
                    f"{c['sim_fidelity']:.6f}",
                    f"{c['gap']:.6f}",
                    c["within_5pct"],
                ])
        print(f"  Saved: {csv_path2}")


def save_results_json(hw_results, comparison, output_dir):
    """Save full results as JSON for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {
        "backend": hw_results["backend"],
        "shots": hw_results["shots"],
        "num_qubits_backend": hw_results["num_qubits_backend"],
        "mean_fidelity": hw_results["mean_fidelity"],
        "std_fidelity": hw_results["std_fidelity"],
        "job_id": hw_results["job_id"],
        "results": [],
    }
    for r in hw_results["results"]:
        results_serializable["results"].append({
            "input_name": r["input_name"],
            "fidelity": r["fidelity"],
            "purity": r["purity"],
            "tomo_counts": {b: {str(k): v for k, v in counts.items()}
                            for b, counts in r["tomo_counts"].items()},
            "density_matrix_real": r["density_matrix"].real.tolist(),
            "density_matrix_imag": r["density_matrix"].imag.tolist(),
        })

    json_path = os.path.join(output_dir, "hardware_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"  Saved: {json_path}")

    if comparison:
        comp_serializable = {
            "mean_gap": comparison["mean_gap"],
            "max_gap": comparison["max_gap"],
            "all_within_5pct": comparison["all_within_5pct"],
            "hw_mean": comparison["hw_mean"],
            "sim_mean": comparison["sim_mean"],
            "comparisons": comparison["comparisons"],
        }
        json_path2 = os.path.join(output_dir, "hardware_comparison.json")
        with open(json_path2, "w", encoding="utf-8") as f:
            json.dump(comp_serializable, f, indent=2)
        print(f"  Saved: {json_path2}")


def plot_comparison(hw_results, comparison, output_dir):
    """Generate hardware vs simulation comparison figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    os.makedirs(output_dir, exist_ok=True)

    states = [c["state"] for c in comparison["comparisons"]]
    hw_fids = [c["hw_fidelity"] for c in comparison["comparisons"]]
    sim_fids = [c["sim_fidelity"] for c in comparison["comparisons"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Bar chart comparison
    ax = axes[0]
    x = np.arange(len(states))
    width = 0.35
    bars_hw = ax.bar(x - width/2, hw_fids, width, label='Hardware',
                     color='#2196F3', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars_sim = ax.bar(x + width/2, sim_fids, width, label='Simulation',
                      color='#FF9800', alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Fidelity', fontsize=14)
    ax.set_title(f'Hardware vs Simulation ({hw_results["backend"]})', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(states, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars_hw:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars_sim:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    # Panel B: Gap analysis
    ax2 = axes[1]
    gaps = [c["gap"] for c in comparison["comparisons"]]
    colors = ['#4CAF50' if abs(g) < 0.05 else '#F44336' for g in gaps]
    bars = ax2.bar(x, gaps, width=0.6, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.axhline(y=0.05, color='gray', linewidth=0.8, linestyle='--',
                label='+5% threshold')
    ax2.axhline(y=-0.05, color='gray', linewidth=0.8, linestyle='--',
                label='-5% threshold')
    ax2.set_ylabel('Fidelity Gap (HW - Sim)', fontsize=14)
    ax2.set_title('Hardware-Simulation Gap', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(states, fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "fig11_hw_vs_simulation.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Hardware validation")
    parser.add_argument("--backend", default="ibm_torino",
                        help="IBM backend name")
    parser.add_argument("--shots", type=int, default=8192,
                        help="Shots per circuit")
    parser.add_argument("--opt-level", type=int, default=3,
                        help="Transpiler optimization level")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory")
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip simulation comparison")
    args = parser.parse_args()

    print("=" * 60)
    print("  QUANTUM TELEPORTATION — HARDWARE VALIDATION")
    print("=" * 60)
    t_start = time.time()

    # ── Step 1: Run on hardware ─────────────────────────────
    print(f"\n[Step 1] Running on {args.backend} ({args.shots} shots)")
    hw_results = run_hardware_validation(
        backend_name=args.backend,
        shots=args.shots,
        optimization_level=args.opt_level,
        verbose=True,
    )

    # ── Step 2: Compare with simulation ─────────────────────
    comparison = None
    if not args.skip_sim:
        print(f"\n[Step 2] Comparing with simulation results")
        try:
            from src.teleportation.protocol_executor import TeleportationProtocolSimulator
            from src.noise_models.composite_noise import ComprehensiveTeleportationNoise
            from src.utils.device_parameters import load_device_parameters

            # Use ibmq_manila parameters (default simulation device)
            params = load_device_parameters("ibmq_manila")
            noise_model = ComprehensiveTeleportationNoise(params)
            protocol_sim = TeleportationProtocolSimulator(noise_model, params)
            sim_results = protocol_sim.simulate_all_test_states(verbose=False)
            comparison = compare_hardware_simulation(hw_results, sim_results, verbose=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  Warning: Could not run simulation comparison: {e}")
            print("  Hardware results will be saved without comparison.")

    # ── Step 3: Save results ────────────────────────────────
    print(f"\n[Step 3] Saving results")
    save_results_csv(hw_results, comparison, args.output_dir)
    save_results_json(hw_results, comparison, args.output_dir)

    # ── Step 4: Generate figure ─────────────────────────────
    if comparison:
        print(f"\n[Step 4] Generating comparison figure")
        plot_comparison(hw_results, comparison, args.output_dir)

    # ── Summary ─────────────────────────────────────────────
    t_total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  HARDWARE VALIDATION COMPLETE")
    print(f"  Backend:        {hw_results['backend']}")
    print(f"  Job ID:         {hw_results['job_id']}")
    print(f"  HW Mean F:      {hw_results['mean_fidelity']:.4f} ± "
          f"{hw_results['std_fidelity']:.4f}")
    if comparison:
        print(f"  Sim Mean F:     {comparison['sim_mean']:.4f}")
        print(f"  Mean |gap|:     {comparison['mean_gap']:.4f}")
        print(f"  Validation:     "
              f"{'PASSED' if comparison['mean_gap'] < 0.05 else 'PARTIAL' if comparison['mean_gap'] < 0.10 else 'REVIEW'}")
    print(f"  Time:           {t_total:.0f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
