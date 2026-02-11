import pytest
import numpy as np

from anytimes.rao import compute_rao


def test_compute_rao_tracks_gain_and_phase_near_target_frequency():
    dt = 0.1
    t = np.arange(0.0, 512.0, dt)
    f0 = 0.2
    gain = 2.5
    phase_deg = 35.0

    excitation = np.sin(2 * np.pi * f0 * t)
    response = gain * np.sin(2 * np.pi * f0 * t + np.deg2rad(phase_deg))

    freqs, amp, phase, coh = compute_rao(excitation, response, dt=dt, nperseg=2048)

    idx = int(np.argmin(np.abs(freqs - f0)))
    assert amp[idx] == pytest.approx(gain, rel=0.08)
    assert phase[idx] == pytest.approx(phase_deg, abs=8.0)
    assert coh[idx] > 0.95


def test_compute_rao_validates_shapes_and_length():
    x = np.ones(16)
    y = np.ones(12)
    with pytest.raises(ValueError, match="same length"):
        compute_rao(x, y, dt=0.1)
