"""Check that short sounds are aligned causally, never stretched over responses."""
import unittest
import numpy as np
from bat_internship.audio_features_legacy import (
    aligned_spectrogram, aligned_mel_spectrogram, mel_filterbank)


class MelTests(unittest.TestCase):
    def test_tone_silence_offset_and_amplitude(self):
        fs = 200000
        tone = np.sin(2 * np.pi * 40000 * np.arange(6400) / fs)
        a, centers = aligned_mel_spectrogram(tone, fs, [-32, 0, 32, 64])
        self.assertEqual(a.shape, (32, 4))
        np.testing.assert_array_equal(a[:, [0, 1, 3]], 0)
        self.assertLess(abs(centers[a[:, 2].argmax()] - 40000), 7000)
        quiet, _ = aligned_mel_spectrogram(tone / 2, fs, [-32, 0, 32, 64])
        np.testing.assert_allclose(np.expm1(quiet), np.expm1(a) / 4, rtol=1e-12)
        delayed, _ = aligned_mel_spectrogram(
            tone, fs, [-32, 0, 32, 64], audio_offset_ms=32)
        np.testing.assert_allclose(a[:, 2], delayed[:, 3])
        np.testing.assert_array_equal(delayed[:, :3], 0)

    def test_future_audio_cannot_change_earlier_features(self):
        x = np.sin(2 * np.pi * 20000 * np.arange(12800) / 200000)
        y = x.copy()
        y[6400:] *= .1
        a, _ = aligned_mel_spectrogram(x, 200000, [0, 32, 64])
        b, _ = aligned_mel_spectrogram(y, 200000, [0, 32, 64])
        np.testing.assert_array_equal(a[:, :2], b[:, :2])
        self.assertGreater(a[:, 2].max(), b[:, 2].max())

    def test_filters_and_bad_configuration(self):
        filters, centers = mel_filterbank(200000)
        self.assertEqual(filters.shape, (32, 1025))
        self.assertTrue((np.diff(centers) > 0).all())
        self.assertTrue((filters >= 0).all())
        # Area-normalized triangle integrals should be near one.
        np.testing.assert_allclose(filters.sum(axis=1) * 200000 / 2048,
                                   1, atol=.15)
        for kwargs in [dict(n_mels=0), dict(f_max=100001), dict(f_min=-1),
                       dict(n_fft=511), dict(n_mels=1000), dict(window_ms=3),
                       dict(power_reference=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                aligned_mel_spectrogram(np.ones(400), 200000, [0, 32], **kwargs)


class AudioGridTests(unittest.TestCase):
    def test_short_tone_frequency_and_silent_context(self):
        fs = 200000
        audio = np.sin(2 * np.pi * 40000 * np.arange(6400) / fs)
        features, freq = aligned_spectrogram(audio, fs, [-32, 0, 32, 64, 96])
        np.testing.assert_array_equal(features[:, [0, 1, 3, 4]], 0)
        self.assertGreater(features[:, 2].max(), 0)
        self.assertLess(abs(freq[features[:, 2].argmax()] - 40000), 500)

    def test_audio_offset_moves_sound_without_changing_spectrum(self):
        audio = np.sin(2 * np.pi * 30000 * np.arange(6400) / 200000)
        a, _ = aligned_spectrogram(audio, 200000, [0, 32, 64, 96])
        b, _ = aligned_spectrogram(audio, 200000, [0, 32, 64, 96], audio_offset_ms=32)
        np.testing.assert_allclose(a[:, 1], b[:, 2])
        np.testing.assert_array_equal(b[:, [0, 1, 3]], 0)

    def test_no_future_audio_and_no_per_sound_normalization(self):
        audio = np.sin(2 * np.pi * 20000 * np.arange(12800) / 200000)
        later = audio.copy()
        later[6400:] *= 0.25
        a, _ = aligned_spectrogram(audio, 200000, [0, 32, 64])
        b, _ = aligned_spectrogram(later, 200000, [0, 32, 64])
        np.testing.assert_array_equal(a[:, :2], b[:, :2])
        self.assertLess(b[:, 2].max(), a[:, 2].max())

    def test_bad_grids_and_nan_fail(self):
        for times, kwargs in [([0, 31], {}), ([0, 32], {"window_ms": 3}),
                              ([0, 32], {"audio_offset_ms": 0.0001})]:
            with self.subTest(times=times, kwargs=kwargs), self.assertRaises(ValueError):
                aligned_spectrogram(np.ones(100), 200000, times, **kwargs)
        with self.assertRaises(ValueError):
            aligned_spectrogram([np.nan], 200000, [0, 32])


if __name__ == "__main__":
    unittest.main()
