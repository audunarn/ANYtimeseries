"""Frequency-domain RAO utilities."""

from __future__ import annotations

import numpy as np
from scipy import signal


def compute_rao(
    excitation: np.ndarray,
    response: np.ndarray,
    dt: float,
    nperseg: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a complex RAO from paired excitation/response samples.

    Parameters
    ----------
    excitation
        Input signal (e.g. wave elevation).
    response
        Output signal (e.g. motion component).
    dt
        Constant time step in seconds.
    nperseg
        Segment length for Welch/CSD estimates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Frequency in Hz, RAO amplitude, RAO phase in degrees and
        magnitude-squared coherence.
    """

    x = np.asarray(excitation, dtype=float)
    y = np.asarray(response, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Excitation and response must be 1-D arrays.")
    if x.size != y.size:
        raise ValueError("Excitation and response must have the same length.")
    if x.size < 8:
        raise ValueError("At least 8 samples are required to estimate an RAO.")
    if dt <= 0:
        raise ValueError("Time step dt must be positive.")

    fs = 1.0 / dt
    seg = min(int(nperseg), x.size)
    if seg < 8:
        raise ValueError("nperseg is too small for the selected signal length.")

    x = x - np.mean(x)
    y = y - np.mean(y)

    freqs, p_xx = signal.welch(x, fs=fs, nperseg=seg, detrend="constant")
    _, p_xy = signal.csd(x, y, fs=fs, nperseg=seg, detrend="constant")
    _, coh = signal.coherence(x, y, fs=fs, nperseg=seg, detrend="constant")

    eps = np.finfo(float).eps
    h = p_xy / np.maximum(p_xx, eps)

    amp = np.abs(h)
    phase_deg = np.degrees(np.angle(h))

    return freqs, amp, phase_deg, coh

