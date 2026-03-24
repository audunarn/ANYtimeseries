import pytest
import numpy as np

from anytimes.rao import compute_rao, compute_rao_from_timeseries


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


def test_compute_rao_from_timeseries_recovers_signal_amplitude():
    dt = 0.05
    t = np.arange(0.0, 200.0, dt)
    f0 = 0.35
    amp0 = 4.2
    phase0_deg = 20.0

    response = amp0 * np.sin(2 * np.pi * f0 * t + np.deg2rad(phase0_deg))
    freqs, amp, phase, coh = compute_rao_from_timeseries(response, dt=dt)

    idx = int(np.argmin(np.abs(freqs - f0)))
    assert amp[idx] == pytest.approx(amp0, rel=0.03)
    assert phase[idx] == pytest.approx(phase0_deg - 90.0, abs=4.0)
    assert np.isnan(coh[idx])
