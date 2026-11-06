"""Historical custom features; not used by the native deepSTRF waveform path."""
import numpy as np
from numbers import Integral

def _aligned_power(audio, sample_rate, time_ms, *, dt_ms=32.0,
                   audio_offset_ms=0.0, window_ms=2.0, power_reference=1e-12,
                   n_fft=None):
    """Average short-window PSD within [t-dt, t); pad unavailable audio with zeros.

    Time t is the calcium frame timestamp. Each bin is split into nonoverlapping
    2-ms Hann windows; no window reaches beyond that bin. No interpolation,
    waveform stretching, per-stimulus normalization, or future audio is used.
    Only exact integer sample grids and complete short windows are accepted.
    """
    audio = np.asarray(audio, dtype=np.float64)
    time_ms = np.asarray(time_ms, dtype=float)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Expected a nonempty finite mono waveform")
    if (not np.isfinite([sample_rate, dt_ms, window_ms, power_reference, audio_offset_ms]).all()
            or min(sample_rate, dt_ms, window_ms, power_reference) <= 0):
        raise ValueError("Invalid time, sampling or power parameters")
    if (time_ms.ndim != 1 or not time_ms.size or not np.isfinite(time_ms).all()
            or (len(time_ms) > 1 and not np.allclose(np.diff(time_ms), dt_ms))):
        raise ValueError("Expected a uniform calcium time grid")
    sizes = np.array([dt_ms, window_ms]) * sample_rate / 1000
    if not np.allclose(sizes, np.round(sizes), rtol=0, atol=1e-7):
        raise ValueError("Time bins/windows must contain integer numbers of samples")
    bin_samples, win_samples = np.round(sizes).astype(int)
    if win_samples < 4 or bin_samples % win_samples:
        raise ValueError("Each bin must contain complete short windows")
    starts = (time_ms - dt_ms - audio_offset_ms) * sample_rate / 1000
    if not np.allclose(starts, np.round(starts), rtol=0, atol=1e-7):
        raise ValueError("Audio offset must align to the sample grid")
    starts = np.round(starts).astype(int)
    if n_fft is None:
        n_fft = 1 << (int(win_samples) - 1).bit_length()
    if isinstance(n_fft, bool) or not isinstance(n_fft, Integral) or n_fft < win_samples or n_fft % 2:
        raise ValueError("n_fft must be an even integer at least as large as the window")
    window = np.hanning(win_samples)
    features = []
    for start in starts:
        block = np.zeros(bin_samples, dtype=np.float64)
        lo, hi = max(0, start), min(len(audio), start + bin_samples)
        if hi > lo:
            block[lo - start:hi - start] = audio[lo:hi]
        segments = block.reshape(-1, win_samples)
        power = np.abs(np.fft.rfft(segments * window, n=n_fft, axis=-1)) ** 2
        power /= sample_rate * np.sum(window ** 2)
        power[:, 1:-1] *= 2  # One-sided PSD; leave DC and Nyquist unchanged.
        features.append(power.mean(axis=0))
    return np.stack(features, axis=1), np.fft.rfftfreq(n_fft, 1 / sample_rate)


def aligned_spectrogram(audio, sample_rate, time_ms, *, dt_ms=32.0,
                        audio_offset_ms=0.0, window_ms=2.0, power_reference=1e-12):
    """Historical linear-frequency transform, retained for reproducibility only."""
    power, frequencies = _aligned_power(
        audio, sample_rate, time_ms, dt_ms=dt_ms, audio_offset_ms=audio_offset_ms,
        window_ms=window_ms, power_reference=power_reference)
    return np.log1p(power / power_reference), frequencies


def mel_filterbank(sample_rate, *, n_fft=2048, n_mels=32, f_min=0.0, f_max=None):
    """HTK-spaced triangular filters with unit-area (Slaney) normalization.

    Matches the convention librosa.filters.mel(htk=True, norm='slaney').
    The Mel scale is a feature choice, not a measured bat hearing model.
    """
    f_max = sample_rate / 2 if f_max is None else f_max
    if (not np.isfinite([sample_rate, f_min, f_max]).all()
            or not 0 <= f_min < f_max <= sample_rate / 2):
        raise ValueError("Require 0 <= f_min < f_max <= Nyquist")
    if (isinstance(n_mels, bool) or not isinstance(n_mels, Integral) or n_mels < 1
            or isinstance(n_fft, bool) or not isinstance(n_fft, Integral)
            or n_fft < 4 or n_fft % 2):
        raise ValueError("Positive integer n_mels and even n_fft >= 4 required")
    mel_bounds = 2595 * np.log10(1 + np.array([f_min, f_max]) / 700)
    edges = 700 * (10 ** (np.linspace(*mel_bounds, int(n_mels) + 2) / 2595) - 1)
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    lower = (frequencies[None, :] - edges[:-2, None]) / np.diff(edges)[:-1, None]
    upper = (edges[2:, None] - frequencies[None, :]) / np.diff(edges)[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    if np.any(weights.max(axis=1) == 0):
        raise ValueError("Empty Mel bands: increase n_fft or reduce n_mels")
    weights *= (2 / (edges[2:] - edges[:-2]))[:, None]
    return weights, edges[1:-1]


def aligned_mel_spectrogram(audio, sample_rate, time_ms, *, dt_ms=32.0,
                            audio_offset_ms=0.0, window_ms=2.0, n_fft=2048,
                            n_mels=32, f_min=0.0, f_max=None, power_reference=1e-12):
    """Compute (Mel bands, calcium frames) without retaining derived features.

    Hann-window PSDs are averaged inside each preceding calcium bin, projected
    onto area-normalized Mel filters, then transformed with log1p(x/reference).
    FFT zero-padding samples the spectrum more finely; it does not improve the
    physical frequency resolution of the 2-ms window. No per-sound scaling.
    """
    weights, centers = mel_filterbank(sample_rate, n_fft=n_fft, n_mels=n_mels,
                                     f_min=f_min, f_max=f_max)
    power, _ = _aligned_power(audio, sample_rate, time_ms, dt_ms=dt_ms,
        audio_offset_ms=audio_offset_ms, window_ms=window_ms,
        power_reference=power_reference, n_fft=n_fft)
    return np.log1p((weights @ power) / power_reference), centers


