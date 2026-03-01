"""
1/f noise model (non-Markovian dephasing).

Charge and flux noise in superconducting qubits exhibit a 1/f^alpha
power spectral density. This causes non-Markovian dephasing that cannot
be fully captured by a simple T2 model. The 1/f noise leads to
Gaussian dephasing with a characteristic T_phi that depends on the
noise amplitude and spectral exponent.

For a single teleportation protocol execution, we model this as
an additional dephasing contribution beyond the Markovian T2 model.

References:
    - Yan et al. (2016) Nature Communications
    - Ithier et al. (2005) PRB
    - Paladino et al. (2014) Rev. Mod. Phys.
"""

import numpy as np
from qiskit_aer.noise.errors import phase_damping_error


def generate_1f_noise_trajectory(duration_s, dt_s, alpha=0.9, amplitude=5e-6):
    """
    Generate a 1/f^alpha noise time series.

    Parameters
    ----------
    duration_s : float
        Total duration in seconds.
    dt_s : float
        Time step in seconds.
    alpha : float
        Spectral exponent (typically 0.7-1.1 for charge noise).
    amplitude : float
        Noise amplitude in energy units.

    Returns
    -------
    np.ndarray
        Noise trajectory (frequency fluctuation vs time).
    """
    n_steps = max(int(duration_s / dt_s), 2)
    white_noise = np.random.randn(n_steps)

    freqs = np.fft.fftfreq(n_steps, dt_s)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # avoid division by zero

    # 1/f^alpha filter
    psd_filter = 1.0 / np.abs(freqs) ** (alpha / 2.0)
    noise_fft = np.fft.fft(white_noise) * psd_filter
    noise = np.real(np.fft.ifft(noise_fft)) * amplitude

    return noise


def one_over_f_dephasing_rate(protocol_duration_s, alpha=0.9, amplitude=5e-6):
    """
    Estimate additional dephasing from 1/f noise beyond Markovian T2.

    The 1/f noise contribution to dephasing goes as:
        Gamma_1f ~ A^2 * ln(t_protocol / t_IR)

    where t_IR is an infrared cutoff (typically the repetition time).

    For the teleportation protocol (~3 us), this adds a small but
    non-negligible dephasing contribution.

    Parameters
    ----------
    protocol_duration_s : float
        Total protocol duration in seconds.
    alpha : float
        Spectral exponent.
    amplitude : float
        Noise amplitude.

    Returns
    -------
    float
        Additional dephasing probability.
    """
    # Infrared cutoff (repetition time ~ 100 us)
    t_ir = 100e-6
    if protocol_duration_s <= 0:
        return 0.0

    # Gaussian dephasing from 1/f noise
    # Gamma ~ A^2 * (t/t_ir)^(1-alpha) for alpha near 1
    ratio = protocol_duration_s / t_ir
    if ratio <= 0:
        return 0.0

    gamma_1f = amplitude**2 * abs(np.log(ratio + 1e-30)) * 1e6
    # Convert to a dephasing probability (clamp to [0, 1])
    p_dephasing = min(1.0 - np.exp(-gamma_1f), 1.0)
    p_dephasing = max(p_dephasing, 0.0)
    return p_dephasing


def build_1f_dephasing_error(protocol_duration_s, alpha=0.9, amplitude=5e-6):
    """
    Build a phase damping error representing 1/f noise contribution.

    This is added on top of the Markovian T2 dephasing.

    Returns
    -------
    QuantumError or None
        Phase damping error, or None if contribution is negligible.
    """
    p = one_over_f_dephasing_rate(protocol_duration_s, alpha, amplitude)
    if p < 1e-8:
        return None
    return phase_damping_error(p)
