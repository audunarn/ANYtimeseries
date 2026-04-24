"""Frequency-domain RAO utilities.

The main RAO convention used here is:

    H(f) = Y(f) / X(f)

where X is the excitation signal and Y is the response signal.

For paired excitation/response signals, the RAO is estimated using Welch
auto-spectral density and cross-spectral density:

    H(f) = Pxy(f) / Pxx(f)

where SciPy's csd(x, y) convention gives Pxy = conj(X) * Y.

For single-series fallback mode, no true excitation signal is available.
The function therefore returns a one-sided amplitude/phase spectrum of the
response signal and assumes unit excitation amplitude. This is useful as a
last-resort approximation, but it is not a true excitation/response RAO.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def _as_1d_finite_array(values: np.ndarray, name: str) -> np.ndarray:
    """Convert input to a finite 1-D float array."""

    arr = np.asarray(values, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")

    if arr.size < 8:
        raise ValueError(f"{name} must contain at least 8 samples.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return arr


def _validate_dt(dt: float) -> float:
    """Validate and return time step as float."""

    dt = float(dt)

    if not np.isfinite(dt):
        raise ValueError("Time step dt must be finite.")

    if dt <= 0.0:
        raise ValueError("Time step dt must be positive.")

    return dt


def _validate_nperseg(nperseg: int, n_samples: int) -> int:
    """Validate Welch segment length."""

    try:
        seg_requested = int(nperseg)
    except TypeError as exc:
        raise ValueError("nperseg must be an integer.") from exc

    if seg_requested <= 0:
        raise ValueError("nperseg must be positive.")

    seg = min(seg_requested, int(n_samples))

    if seg < 8:
        raise ValueError("nperseg is too small for the selected signal length.")

    return seg


def _remove_mean(values: np.ndarray) -> np.ndarray:
    """Return demeaned copy of the input array."""

    return values - np.mean(values)


def _remove_zero_frequency(
    freqs: np.ndarray,
    amp: np.ndarray,
    phase_deg: np.ndarray,
    coh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove the zero-frequency/DC component."""

    valid = freqs > 0.0

    return freqs[valid], amp[valid], phase_deg[valid], coh[valid]


def compute_rao(
    excitation: np.ndarray,
    response: np.ndarray,
    dt: float,
    nperseg: int = 1024,
    *,
    remove_zero_frequency: bool = False,
    unwrap_phase: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a complex RAO from paired excitation/response samples.

    The estimated RAO is:

        H(f) = Y(f) / X(f)

    where X is the excitation and Y is the response.

    Parameters
    ----------
    excitation
        Input signal, for example wave elevation.
    response
        Output signal, for example motion, force, acceleration, or another
        response component.
    dt
        Constant time step in seconds.
    nperseg
        Segment length used by Welch/CSD estimates. If larger than the signal
        length, the full signal length is used.
    remove_zero_frequency
        If True, remove the DC component from the returned arrays.
    unwrap_phase
        If True, unwrap phase before converting to degrees. If False, phase is
        returned wrapped in the range [-180, 180].

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Frequency in Hz, RAO amplitude, RAO phase in degrees, and
        magnitude-squared coherence.

    Notes
    -----
    Phase is the response phase relative to the excitation.

    The estimator uses SciPy's convention:

        csd(x, y) = conj(X) * Y

    Therefore:

        H(f) = Pxy(f) / Pxx(f)

    gives response divided by excitation.
    """

    x = _as_1d_finite_array(excitation, "Excitation")
    y = _as_1d_finite_array(response, "Response")
    dt = _validate_dt(dt)

    if x.size != y.size:
        raise ValueError("Excitation and response must have the same length.")

    seg = _validate_nperseg(nperseg, x.size)

    fs = 1.0 / dt

    x = _remove_mean(x)
    y = _remove_mean(y)

    freqs, p_xx = signal.welch(
        x,
        fs=fs,
        nperseg=seg,
        detrend="constant",
    )

    _, p_xy = signal.csd(
        x,
        y,
        fs=fs,
        nperseg=seg,
        detrend="constant",
    )

    _, coh = signal.coherence(
        x,
        y,
        fs=fs,
        nperseg=seg,
        detrend="constant",
    )

    eps = np.finfo(float).eps
    h = p_xy / np.maximum(p_xx, eps)

    amp = np.abs(h)

    if unwrap_phase:
        phase_deg = np.degrees(np.unwrap(np.angle(h)))
    else:
        phase_deg = np.degrees(np.angle(h))

    if remove_zero_frequency:
        freqs, amp, phase_deg, coh = _remove_zero_frequency(
            freqs,
            amp,
            phase_deg,
            coh,
        )

    return freqs, amp, phase_deg, coh


def compute_rao_from_timeseries(
    response: np.ndarray,
    dt: float,
    *,
    remove_zero_frequency: bool = False,
    unwrap_phase: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a fallback single-series RAO-like spectrum.

    This function is intended for the worst-case situation where no excitation
    signal is available.

    It assumes unit excitation amplitude and returns the one-sided FFT
    amplitude/phase spectrum of the response signal.

    Parameters
    ----------
    response
        Response time series.
    dt
        Constant time step in seconds.
    remove_zero_frequency
        If True, remove the DC component from the returned arrays.
    unwrap_phase
        If True, unwrap phase before converting to degrees. If False, phase is
        returned wrapped in the range [-180, 180].

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Frequency in Hz, amplitude, phase in degrees, and coherence.

        Coherence is returned as NaN because no paired excitation signal is
        available.

    Notes
    -----
    This is not a true excitation/response RAO.

    The phase is the FFT phase relative to the start of the time record, not
    relative to an excitation signal.
    """

    y = _as_1d_finite_array(response, "Response")
    dt = _validate_dt(dt)

    y = _remove_mean(y)
    n = y.size

    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.fft.rfft(y)

    amp = 2.0 * np.abs(spec) / float(n)

    if amp.size:
        amp[0] *= 0.5

        if n % 2 == 0 and amp.size > 1:
            amp[-1] *= 0.5

    if unwrap_phase:
        phase_deg = np.degrees(np.unwrap(np.angle(spec)))
    else:
        phase_deg = np.degrees(np.angle(spec))

    coh = np.full_like(freqs, np.nan, dtype=float)

    if remove_zero_frequency:
        freqs, amp, phase_deg, coh = _remove_zero_frequency(
            freqs,
            amp,
            phase_deg,
            coh,
        )

    return freqs, amp, phase_deg, coh


def compute_single_sided_amplitude_spectrum(
    signal_values: np.ndarray,
    dt: float,
    *,
    remove_zero_frequency: bool = False,
    unwrap_phase: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a one-sided FFT amplitude/phase spectrum.

    This is a clearer generic alias for the single-series calculation used by
    ``compute_rao_from_timeseries``.

    Parameters
    ----------
    signal_values
        Input signal.
    dt
        Constant time step in seconds.
    remove_zero_frequency
        If True, remove the DC component from the returned arrays.
    unwrap_phase
        If True, unwrap phase before converting to degrees. If False, phase is
        returned wrapped in the range [-180, 180].

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Frequency in Hz, one-sided amplitude, and phase in degrees.
    """

    freqs, amp, phase_deg, _ = compute_rao_from_timeseries(
        response=signal_values,
        dt=dt,
        remove_zero_frequency=remove_zero_frequency,
        unwrap_phase=unwrap_phase,
    )

    return freqs, amp, phase_deg