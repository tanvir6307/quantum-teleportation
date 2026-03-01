"""
Data export framework.

Exports all simulation results to CSV files for reproducibility.
Every figure in the study can be reproduced from these CSV files
without re-running simulations.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime


class DataExporter:
    """Centralized data export manager."""

    def __init__(self, base_dir="data/simulation_results"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.exported_files = []
        self.metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "numpy_version": np.__version__,
        }

    def _save(self, df, subdir, filename):
        """Save DataFrame to CSV."""
        dirpath = os.path.join(self.base_dir, subdir)
        os.makedirs(dirpath, exist_ok=True)
        filepath = os.path.join(dirpath, filename)
        df.to_csv(filepath, index=False)
        self.exported_files.append(filepath)
        print(f"  Exported: {filepath}")
        return filepath

    def export_teleportation_fidelity(self, results):
        """Export teleportation fidelity results for all test states."""
        rows = []
        for r in results:
            rows.append({
                "input_state": r["input_name"],
                "fidelity": r["fidelity"],
                "purity": r["purity"],
                "simulation_time_s": r["simulation_time_s"],
                "above_classical_limit": r["fidelity"] > 2/3,
            })
        df = pd.DataFrame(rows)
        return self._save(df, "teleportation_fidelity", "all_states_fidelity.csv")

    def export_error_budget(self, error_budget):
        """Export error budget breakdown."""
        rows = []
        for source, data in error_budget.items():
            rows.append({
                "error_source": source,
                "contribution_percent": data["percent"],
                "contribution_absolute": data["absolute"],
                "phase": data.get("phase", ""),
                "notes": data.get("notes", ""),
            })
        df = pd.DataFrame(rows)
        return self._save(df, "error_budget", "error_budget.csv")

    def export_bell_preparation(self, bell_data):
        """Export Bell pair error accumulation data."""
        steps = bell_data["steps"]
        df = pd.DataFrame(steps)
        return self._save(df, "bell_pair_preparation", "bell_error_accumulation.csv")

    def export_noise_comparison(self, comparison_data):
        """Export noise model comparison."""
        rows = [{"noise_model": k, "fidelity": v}
                for k, v in comparison_data.items()]
        df = pd.DataFrame(rows)
        return self._save(df, "noise_comparison", "noise_model_comparison.csv")

    def export_protocol_timeline(self, phases, total_time_ns):
        """Export protocol timeline."""
        rows = []
        for p in phases:
            rows.append({
                "phase_number": p["phase_number"],
                "phase_name": p["phase_name"],
                "start_time_ns": p["start_time_ns"],
                "end_time_ns": p["end_time_ns"],
                "duration_ns": p["duration_ns"],
                "active_qubits": str(p["active_qubits"]),
                "idle_qubits": str(p["idle_qubits"]),
                "noise_sources": str(p["noise_sources"]),
            })
        df = pd.DataFrame(rows)
        return self._save(df, "protocol_timeline", "timeline.csv")

    def export_parameter_sweep(self, param_name, param_values, fidelities):
        """Export parameter sweep results."""
        df = pd.DataFrame({
            "parameter_name": [param_name] * len(param_values),
            "parameter_value": param_values,
            "fidelity": fidelities,
        })
        return self._save(df, "parameter_sweeps",
                         f"fidelity_vs_{param_name}.csv")

    def export_heatmap(self, x_name, x_vals, y_name, y_vals, z_matrix):
        """Export 2D heatmap data."""
        rows = []
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                rows.append({
                    "x_param": x_name, "x_value": x,
                    "y_param": y_name, "y_value": y,
                    "fidelity": z_matrix[j, i],
                })
        df = pd.DataFrame(rows)
        return self._save(df, "parameter_sweeps", "fidelity_heatmap_t1_t2.csv")

    def export_monte_carlo(self, trials_list, mean_fids, std_errs):
        """Export Monte Carlo convergence data."""
        df = pd.DataFrame({
            "num_trials": trials_list,
            "mean_fidelity": mean_fids,
            "std_error": std_errs,
            "ci_95_lower": [m - 1.96*s for m, s in zip(mean_fids, std_errs)],
            "ci_95_upper": [m + 1.96*s for m, s in zip(mean_fids, std_errs)],
            "converged": [s < 0.01 for s in std_errs],
        })
        return self._save(df, "statistical_analysis", "monte_carlo_convergence.csv")

    def export_zne_results(self, noise_scales, fidelities, extrapolated):
        """Export ZNE results."""
        df = pd.DataFrame({
            "noise_scale": list(noise_scales) + [0.0],
            "fidelity": list(fidelities) + [extrapolated],
            "type": ["measured"] * len(noise_scales) + ["extrapolated"],
        })
        return self._save(df, "error_mitigation", "zero_noise_extrapolation.csv")

    def export_cumulative_decay(self, trajectory):
        """Export cumulative fidelity decay trajectory."""
        df = pd.DataFrame(trajectory)
        return self._save(df, "error_budget", "cumulative_fidelity_decay.csv")

    def export_metadata(self, params, sim_config):
        """Export simulation metadata."""
        meta = {
            **self.metadata,
            "device_parameters": {k: v for k, v in params.items()
                                   if not isinstance(v, np.ndarray)},
            "simulation_config": sim_config,
            "total_exported_files": len(self.exported_files),
        }
        dirpath = os.path.join(os.path.dirname(self.base_dir), "metadata")
        os.makedirs(dirpath, exist_ok=True)
        filepath = os.path.join(dirpath, "simulation_metadata.json")

        # Make JSON serializable
        def make_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        clean_meta = json.loads(
            json.dumps(meta, default=make_serializable)
        )
        with open(filepath, "w") as f:
            json.dump(clean_meta, f, indent=2)
        self.exported_files.append(filepath)
        print(f"  Exported: {filepath}")

    def summary(self):
        """Print export summary."""
        print(f"\n{'='*50}")
        print(f"DATA EXPORT SUMMARY")
        print(f"{'='*50}")
        print(f"Total files exported: {len(self.exported_files)}")
        for f in self.exported_files:
            print(f"  {f}")
        print(f"{'='*50}")
