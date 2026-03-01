"""
Statistical analysis module for simulation validation.

Provides rigorous statistical tests to compare simulated results
against experimental benchmarks, including:
- Z-test for fidelity comparisons
- Chi-square goodness-of-fit for error budget distributions
- Bootstrap confidence intervals
- Cohen's d effect size
- Kolmogorov-Smirnov test for distribution comparisons
"""

import numpy as np
from scipy import stats


def z_test_fidelity(sim_fidelity, exp_fidelity, exp_error, sim_error=None):
    """
    Two-sample Z-test for comparing simulated and experimental fidelity.

    H0: F_sim = F_exp
    H1: F_sim ≠ F_exp

    Parameters
    ----------
    sim_fidelity : float
        Simulated fidelity.
    exp_fidelity : float
        Experimental fidelity.
    exp_error : float
        Standard error of experimental measurement.
    sim_error : float, optional
        Standard error of simulation (0 for deterministic).

    Returns
    -------
    dict
        z_statistic, p_value, significant_at_005, significant_at_001
    """
    sigma_sim = sim_error if sim_error else 0.0
    sigma_total = np.sqrt(exp_error**2 + sigma_sim**2)

    if sigma_total < 1e-15:
        z = float('inf') if abs(sim_fidelity - exp_fidelity) > 0 else 0.0
    else:
        z = (sim_fidelity - exp_fidelity) / sigma_total

    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "z_statistic": z,
        "p_value": p_value,
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
        "sim_fidelity": sim_fidelity,
        "exp_fidelity": exp_fidelity,
        "exp_error": exp_error,
        "gap": abs(sim_fidelity - exp_fidelity),
    }


def chi_square_error_budget(sim_budget, exp_budget):
    """
    Chi-square goodness-of-fit test comparing simulated vs. expected
    error budget distributions.

    Parameters
    ----------
    sim_budget : dict
        Simulated error budget {source: percent}.
    exp_budget : dict
        Expected error budget {source: percent}.

    Returns
    -------
    dict
        chi2_statistic, p_value, degrees_of_freedom
    """
    # Align keys
    common_keys = sorted(set(sim_budget.keys()) & set(exp_budget.keys()))
    if not common_keys:
        return {"chi2_statistic": float('nan'), "p_value": float('nan'),
                "degrees_of_freedom": 0, "note": "No common error sources"}

    observed = np.array([sim_budget[k] for k in common_keys])
    expected = np.array([exp_budget[k] for k in common_keys])

    # Normalize to same total
    if np.sum(expected) > 0:
        expected = expected / np.sum(expected) * np.sum(observed)

    # Avoid zero expected values
    mask = expected > 0.01
    if np.sum(mask) < 2:
        return {"chi2_statistic": float('nan'), "p_value": float('nan'),
                "degrees_of_freedom": 0, "note": "Insufficient non-zero categories"}

    chi2, p_value = stats.chisquare(observed[mask], expected[mask])

    return {
        "chi2_statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": int(np.sum(mask)) - 1,
        "categories": [common_keys[i] for i, m in enumerate(mask) if m],
    }


def bootstrap_confidence_interval(fidelities, n_bootstrap=10000,
                                   confidence_level=0.95, seed=42):
    """
    Bootstrap confidence interval for mean fidelity.

    Parameters
    ----------
    fidelities : array-like
        Array of fidelity measurements.
    n_bootstrap : int
        Number of bootstrap samples.
    confidence_level : float
        Confidence level (0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        mean, ci_lower, ci_upper, std, n_samples
    """
    rng = np.random.default_rng(seed)
    fidelities = np.array(fidelities)
    n = len(fidelities)

    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(fidelities[idx])

    alpha = 1 - confidence_level
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        "mean": np.mean(fidelities),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "std": np.std(fidelities),
        "stderr": np.std(fidelities) / np.sqrt(n),
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
    }


def cohens_d(group1, group2):
    """
    Compute Cohen's d effect size between two groups.

    Parameters
    ----------
    group1, group2 : array-like
        Two groups of measurements.

    Returns
    -------
    dict
        d, interpretation
    """
    g1 = np.array(group1)
    g2 = np.array(group2)

    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

    # Pooled std
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if s_pooled < 1e-15:
        d = 0.0
    else:
        d = (np.mean(g1) - np.mean(g2)) / s_pooled

    # Interpretation
    ad = abs(d)
    if ad < 0.2:
        interpretation = "negligible"
    elif ad < 0.5:
        interpretation = "small"
    elif ad < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "d": d,
        "abs_d": ad,
        "interpretation": interpretation,
    }


def ks_test_fidelity_distribution(sim_fidelities, reference_mean,
                                    reference_std):
    """
    Kolmogorov-Smirnov test: compare simulated fidelity distribution
    against a reference normal distribution.

    Parameters
    ----------
    sim_fidelities : array-like
        Simulated fidelity values.
    reference_mean : float
        Expected mean fidelity.
    reference_std : float
        Expected standard deviation.

    Returns
    -------
    dict
        ks_statistic, p_value
    """
    if reference_std < 1e-15:
        return {"ks_statistic": float('nan'), "p_value": float('nan'),
                "note": "Reference std is zero"}

    sim = np.array(sim_fidelities)
    ks_stat, p_value = stats.kstest(sim, 'norm',
                                     args=(reference_mean, reference_std))

    return {
        "ks_statistic": ks_stat,
        "p_value": p_value,
        "sim_mean": np.mean(sim),
        "sim_std": np.std(sim),
        "ref_mean": reference_mean,
        "ref_std": reference_std,
        "consistent": p_value > 0.05,
    }


def validate_against_benchmarks(sim_fidelity, sim_error=None):
    """
    Compare simulated teleportation fidelity against all experimental
    benchmarks using Z-tests.

    Parameters
    ----------
    sim_fidelity : float
        Mean simulated fidelity.
    sim_error : float, optional
        Simulation standard error.

    Returns
    -------
    dict
        {benchmark_name: z_test_result}
    """
    benchmarks = {
        "IBM_hardware_typical": {
            "fidelity": 0.73,
            "error": 0.03,
            "source": "IBM Quantum calibration data (2022-2024)",
        },
        "Steffen_2013_SC": {
            "fidelity": 0.78,
            "error": 0.02,
            "source": "Steffen et al. (2013) Nature",
        },
        "Zhao_2021_SC": {
            "fidelity": 0.76,
            "error": 0.03,
            "source": "Zhao et al. (2021) QST",
        },
        "Research_plan_predicted": {
            "fidelity": 0.757,
            "error": 0.04,
            "source": "Analytical estimate from research plan",
        },
        "Valivarthi_2020_photonic": {
            "fidelity": 0.90,
            "error": 0.03,
            "source": "Valivarthi et al. (2020) PRX Quantum",
        },
        "Quantinuum_2024_logical": {
            "fidelity": 0.975,
            "error": 0.002,
            "source": "Acharya et al. (2024) Science",
        },
    }

    results = {}
    for name, data in benchmarks.items():
        z_result = z_test_fidelity(sim_fidelity, data["fidelity"],
                                    data["error"], sim_error)
        z_result["source"] = data["source"]
        results[name] = z_result

    return results


def format_statistical_report(benchmark_results, boot_ci=None):
    """
    Format a human-readable statistical report.

    Returns
    -------
    str
    """
    lines = []
    lines.append("=" * 70)
    lines.append("STATISTICAL VALIDATION REPORT")
    lines.append("=" * 70)

    if boot_ci:
        lines.append(f"\n  Bootstrap {boot_ci['confidence_level']*100:.0f}% CI: "
                      f"[{boot_ci['ci_lower']:.4f}, {boot_ci['ci_upper']:.4f}]")
        lines.append(f"  Mean: {boot_ci['mean']:.4f} ± {boot_ci['stderr']:.4f} "
                      f"(n={boot_ci['n_samples']}, "
                      f"bootstrap n={boot_ci['n_bootstrap']})")

    lines.append(f"\n  {'Benchmark':<30s} {'Exp':>6s} {'Sim':>6s} "
                 f"{'Gap':>6s} {'z':>7s} {'p':>8s} {'Sig?':>5s}")
    lines.append("  " + "-" * 68)

    for name, res in benchmark_results.items():
        sig = "***" if res["significant_at_001"] else (
              "**" if res["significant_at_005"] else "ns")
        lines.append(
            f"  {name:<30s} {res['exp_fidelity']:>6.3f} "
            f"{res['sim_fidelity']:>6.3f} {res['gap']:>6.3f} "
            f"{res['z_statistic']:>7.2f} {res['p_value']:>8.4f} {sig:>5s}"
        )

    lines.append("")
    lines.append("  Significance: *** p<0.01, ** p<0.05, ns = not significant")
    lines.append("  'ns' against a benchmark means simulation is consistent with it")
    lines.append("=" * 70)

    return "\n".join(lines)
