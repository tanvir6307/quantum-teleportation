"""
Publication-quality visualization suite for quantum teleportation study.

Generates all figures from simulation data (or CSV files for reproducibility).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Set publication-quality defaults
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.figsize": (8, 5),
})


def plot_error_budget_pie(error_budget, save_path=None):
    """
    Figure 1: Error budget breakdown pie chart.
    """
    labels = []
    sizes = []
    for source, data in error_budget.items():
        if source == "TOTAL_infidelity_estimate":
            continue
        if data["percent"] > 0.01:  # Only show non-trivial contributions
            labels.append(source.replace("_", " "))
            sizes.append(data["percent"])

    # Sort by size descending for better visual layout
    paired = sorted(zip(sizes, labels), reverse=True)
    sizes = [p[0] for p in paired]
    labels = [p[1] for p in paired]

    colors = sns.color_palette("Set2", len(labels))

    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%",
        startangle=90, colors=colors, pctdistance=0.80,
        textprops={"fontsize": 13, "fontweight": "bold"},
    )
    for text in autotexts:
        text.set_fontsize(13)
        text.set_fontweight("bold")

    # Use a legend for clear labeling (avoids overlap/clipping)
    legend_labels = [f"{lab} ({sz:.1f}%)" for lab, sz in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title="Error Sources",
              loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=13, title_fontsize=14, frameon=True,
              fancybox=True, shadow=True)

    ax.set_title("Teleportation Error Budget Breakdown\n(Contribution to Total Infidelity)",
                 fontsize=18, fontweight="bold", pad=20)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_fidelity_by_state(results, save_path=None):
    """
    Figure 2: Teleportation fidelity for each input state.
    """
    names = [r["input_name"] for r in results]
    fidelities = [r["fidelity"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, fidelities, color=sns.color_palette("viridis", len(names)),
                  edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Input State")
    ax.set_ylabel("Teleportation Fidelity")
    ax.set_title("Quantum Teleportation Fidelity by Input State\n(Comprehensive Noise Model)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Annotate fidelity values
    for bar, fid in zip(bars, fidelities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{fid:.3f}", ha="center", va="bottom", fontsize=13)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_noise_model_comparison(comparison_data, save_path=None):
    """
    Figure 3: Comparison of different noise model configurations.
    """
    models = list(comparison_data.keys())
    fidelities = list(comparison_data.values())

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, fidelities, color=colors[:len(models)],
                  edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Noise Model Configuration")
    ax.set_ylabel("Teleportation Fidelity")
    ax.set_title("Impact of Different Noise Sources on Teleportation Fidelity")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar, fid in zip(bars, fidelities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{fid:.4f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_protocol_timeline(phases, total_time_ns, save_path=None):
    """
    Figure 4: Protocol timeline with phase durations.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = sns.color_palette("Set2", len(phases))
    y_pos = 0

    for i, phase in enumerate(phases):
        ax.barh(y_pos, phase["duration_ns"],
                left=phase["start_time_ns"],
                height=0.6, color=colors[i],
                edgecolor="black", linewidth=0.5,
                label=phase["phase_name"])
        # Label inside bar — large, vertical text
        mid = phase["start_time_ns"] + phase["duration_ns"] / 2
        ax.text(mid, y_pos, f"{phase['duration_ns']} ns",
                ha="center", va="center", fontsize=14, fontweight="bold",
                rotation=90)

    ax.set_xlabel("Time (ns)", fontsize=16)
    ax.set_yticks([0])
    ax.set_yticklabels(["Protocol"], fontsize=14)
    ax.set_title(f"Teleportation Protocol Timeline (Total: {total_time_ns} ns = {total_time_ns/1e3:.2f} \u03bcs)",
                 fontsize=18, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.15), ncol=3, fontsize=13)
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="x", alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_bell_error_accumulation(bell_data, save_path=None):
    """
    Figure 5: Bell pair fidelity degradation during preparation.
    """
    steps = bell_data["steps"]
    times = [s["time_ns"] for s in steps]
    fids = [s["fidelity"] for s in steps]
    gates = [s["gate"] for s in steps]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, fids, "o-", linewidth=2.5, markersize=12, color="#2c3e50",
            zorder=5)

    # Annotation boxes below points with per-point horizontal offsets
    # Point 0 "none" at x=0: shift right to clear y-axis
    # Point 1 "H(q0)" at x=50: shift much right to clear box 0
    # Point 2 "CNOT" at x=350: centered is fine
    offsets = [(60, -60), (120, -60), (0, -60)]
    for i, (t, f, g) in enumerate(zip(times, fids, gates)):
        ox, oy = offsets[i] if i < len(offsets) else (0, -60)
        ax.annotate(f"{g}\nF = {f:.4f}", (t, f),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=13, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="gray", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel("Time (ns)", fontsize=16)
    ax.set_ylabel("Fidelity", fontsize=16)
    ax.set_title("Bell Pair Fidelity Degradation During Preparation",
                 fontsize=18, fontweight="bold")
    ax.set_ylim(0.95, 1.005)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_parameter_sweep(param_name, param_values, fidelities, save_path=None):
    """
    Figure 6: Fidelity vs. a swept parameter.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_values, fidelities, "o-", linewidth=2, markersize=6, color="#2980b9")

    ax.set_xlabel(param_name)
    ax.set_ylabel("Teleportation Fidelity")
    ax.set_title(f"Teleportation Fidelity vs. {param_name}")
    ax.grid(alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_fidelity_heatmap(t1_values, t2_values, fidelity_matrix, save_path=None):
    """
    Figure 7: 2D heatmap of fidelity vs T1 and T2.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    t1_labels = [f"{t*1e6:.0f}" for t in t1_values]
    t2_labels = [f"{t*1e6:.0f}" for t in t2_values]

    sns.heatmap(fidelity_matrix, annot=True, fmt=".3f",
                xticklabels=t1_labels, yticklabels=t2_labels,
                cmap="RdYlGn", vmin=0.5, vmax=1.0, ax=ax,
                cbar_kws={"label": "Fidelity"})
    ax.set_xlabel("T1 (μs)")
    ax.set_ylabel("T2 (μs)")
    ax.set_title("Teleportation Fidelity vs. Coherence Times")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_monte_carlo_convergence(shots_list, mean_fids, std_errs, save_path=None):
    """
    Figure 8: Monte Carlo convergence analysis.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Mean fidelity vs trials
    ax1.errorbar(shots_list, mean_fids, yerr=[1.96*s for s in std_errs],
                 marker="o", capsize=5, linewidth=2, color="#2c3e50")
    ax1.set_xlabel("Number of Trials")
    ax1.set_ylabel("Mean Fidelity")
    ax1.set_title("Fidelity Convergence")
    ax1.grid(alpha=0.3)

    # Standard error vs trials
    ax2.plot(shots_list, std_errs, "o-", linewidth=2, color="#e74c3c")
    ax2.axhline(y=0.01, color="green", linestyle="--",
                label="Convergence threshold (0.01)")
    ax2.set_xlabel("Number of Trials")
    ax2.set_ylabel("Standard Error")
    ax2.set_title("Sampling Error Convergence")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_cumulative_fidelity_decay(trajectory, save_path=None):
    """
    Figure 9: Cumulative fidelity decay through error sources.
    """
    sources = [t["source"].replace("_", " ") for t in trajectory]
    cum_fids = [t["cumulative_fidelity"] for t in trajectory]
    contributions = [t["infidelity_contribution"] * 100 for t in trajectory]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1]})

    # Upper: cumulative fidelity
    ax1.plot(range(len(sources)), cum_fids, "o-", linewidth=2.5,
             markersize=8, color="#2c3e50")
    ax1.fill_between(range(len(sources)), cum_fids, alpha=0.1, color="#2c3e50")
    ax1.set_xticks(range(len(sources)))
    ax1.set_xticklabels(sources, rotation=45, ha="right")
    ax1.set_ylabel("Cumulative Fidelity")
    ax1.set_title("Cumulative Fidelity Decay Through Error Sources")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0.5, 1.02)

    # Lower: individual contributions
    colors = sns.color_palette("Reds_r", len(sources))
    ax2.bar(range(len(sources)), contributions, color=colors,
            edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(sources)))
    ax2.set_xticklabels(sources, rotation=45, ha="right")
    ax2.set_ylabel("Infidelity Contribution (%)")
    ax2.set_title("Individual Error Source Contributions")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_zne_results(noise_scales, fidelities, extrapolated_fid, save_path=None):
    """
    Figure 10: Zero-noise extrapolation results.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(noise_scales, fidelities, "o", markersize=10, color="#2980b9",
            label="Measured", zorder=5)

    # Fit curve
    coeffs = np.polyfit(noise_scales, fidelities, 2)
    x_fit = np.linspace(0, max(noise_scales) * 1.1, 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, "--", color="#95a5a6", label="Quadratic fit")

    # Extrapolated point
    ax.plot(0, extrapolated_fid, "s", markersize=12, color="#e74c3c",
            label=f"ZNE (F={extrapolated_fid:.4f})", zorder=5)

    ax.set_xlabel("Noise Scale Factor")
    ax.set_ylabel("Teleportation Fidelity")
    ax.set_title("Zero-Noise Extrapolation (ZNE)")
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close()
