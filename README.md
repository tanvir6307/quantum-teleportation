# Predicting Quantum Teleportation Fidelity on Noisy Hardware with Multi-Channel Noise Modeling and Experimental Validation

[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/)
[![Qiskit 2.3](https://img.shields.io/badge/qiskit-2.3.0-6929C4.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A density-matrix simulation framework for quantum teleportation that incorporates **eight physically motivated noise channels** calibrated to IBM superconducting hardware. Unlike conventional circuit simulators that apply the deferred-measurement principle, this framework tracks all **16 branches** of the noisy Bell measurement, explicitly capturing readout misassignment — identified as the single largest fidelity-limiting mechanism (5.2%).

## Key Results

| Metric | Value |
|--------|-------|
| Mean state-transfer fidelity (ibmq_manila) | 0.924 ± 0.012 |
| Process fidelity | 0.886 |
| Multi-device fidelity range | 0.908 – 0.971 |
| Hardware validation gap (ibm_torino) | 2.5% |
| ZNE extrapolated fidelity | 1.000 ± 0.001 |
| χ² goodness-of-fit | p = 0.44 (6 d.o.f.) |

## Noise Channels

The simulation models eight noise sources, each calibrated to IBM Quantum device parameters:

1. **Thermal relaxation** — combined T₁ (energy decay) and T₂ (dephasing)
2. **Depolarizing noise** — single- and two-qubit gate errors
3. **Crosstalk** — static ZZ coupling between nearest-neighbor qubits
4. **Leakage** — population transfer to the |2⟩ state
5. **1/f dephasing** — low-frequency quasi-static flux noise
6. **SPAM errors** — state preparation and readout confusion matrix
7. **Coherent errors** — systematic gate over/under-rotation
8. **Collective dephasing** — spatially correlated phase noise

## Repository Structure

```
quantum_teleportation_simulation/
├── src/
│   ├── teleportation/          # Protocol execution & Bell pair preparation
│   │   ├── bell_preparation.py
│   │   ├── protocol_executor.py
│   │   └── teleportation_circuit.py
│   ├── noise_models/           # Eight noise channel implementations
│   │   ├── composite_noise.py
│   │   ├── markovian_noise.py
│   │   ├── crosstalk_noise.py
│   │   ├── leakage_noise.py
│   │   ├── one_over_f_noise.py
│   │   ├── spam_errors.py
│   │   ├── coherent_errors.py
│   │   └── collective_dephasing.py
│   ├── fidelity/               # State & process fidelity metrics
│   │   └── state_fidelity.py
│   ├── utils/                  # Device parameters, error budget, statistics, visualization
│   │   ├── device_parameters.py
│   │   ├── error_budget.py
│   │   ├── statistical_tests.py
│   │   ├── visualization.py
│   │   └── data_export.py
│   └── validation/             # Hardware validation interface
│       └── hardware_validation.py
├── parameters/
│   └── justified_parameters.json
├── data/                       # Output data (CSV)
│   ├── simulation_results/
│   ├── device_calibration/
│   ├── experimental_benchmarks/
│   └── metadata/
├── figures/                    # Generated publication figures
├── run_simulation.py           # Main simulation pipeline
└── run_hardware_validation.py  # IBM Quantum hardware execution
```


## Quick Start

### Prerequisites

- Python 3.14+
- IBM Quantum account (for hardware validation only)

### Installation

```bash
git clone https://github.com/tanvir6307/quantum-teleportation.git
cd quantum-teleportation
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install qiskit==2.3.0 qiskit-aer==0.17.2 numpy matplotlib
```

### Run the Full Simulation

```bash
cd quantum_teleportation_simulation
python run_simulation.py
```

This executes the complete pipeline:
- Bell pair preparation with fidelity tracking
- Teleportation across six mutually unbiased test states ({|0⟩, |1⟩, |+⟩, |−⟩, |+i⟩, |−i⟩})
- Error budget decomposition
- Noise model hierarchy comparison
- Parameter sweeps (T₁, T₂, CNOT error rate)
- Monte Carlo convergence analysis
- Zero-noise extrapolation
- Multi-device comparison (manila, nairobi, kolkata, torino)
- CSV data export and figure generation

### Hardware Validation (requires IBM Quantum access)

```bash
python run_hardware_validation.py
```

Executes the teleportation protocol on `ibm_torino` (133-qubit Heron r2) with state tomography (8192 shots × 18 circuits).

## Figures

| Figure | Description |
|--------|-------------|
| `fig1_error_budget.png` | Error budget decomposition |
| `fig2_fidelity_by_state.png` | Per-state teleportation fidelity |
| `fig3_noise_comparison.png` | Noise model hierarchy |
| `fig4_protocol_timeline.png` | Protocol phase timeline |
| `fig5_bell_error_accumulation.png` | Bell pair error accumulation |
| `fig6a_fidelity_vs_T1.png` | Fidelity vs. T₁ coherence time |
| `fig6b_fidelity_vs_cnot.png` | Fidelity vs. CNOT error rate |
| `fig7_t1_t2_heatmap.png` | T₁–T₂ fidelity landscape |
| `fig8_monte_carlo.png` | Monte Carlo convergence |
| `fig9_cumulative_decay.png` | Cumulative fidelity decay |
| `fig10_zne.png` | Zero-noise extrapolation |
| `fig11_hw_vs_simulation.png` | Hardware vs. simulation comparison |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tanvir2026teleportation,
    author = {Hassan, Tanvir and Ronggon, Asif Akhtab and Ghose, Pranon and Nurnobi, A. K. M. and Jim, Nur Mohammod and Datta, Aparajita},
    title = {Predicting quantum teleportation fidelity on noisy hardware with multi-channel noise modeling and experimental validation},
    journal = {APL Quantum},
    volume = {3},
    number = {2},
    pages = {026111},
    year = {2026},
    month = {05},
    issn = {2835-0103},
    doi = {10.1063/5.0332767},
    url = {https://doi.org/10.1063/5.0332767},
    eprint = {https://pubs.aip.org/aip/apq/article-pdf/doi/10.1063/5.0332767/21020175/026111_1_5.0332767.pdf},
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

We acknowledge the use of IBM Quantum services. The views expressed are those of the authors and do not reflect the official policy or position of IBM or the IBM Quantum team.
