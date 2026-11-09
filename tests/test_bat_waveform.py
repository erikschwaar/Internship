"""Native waveform grid lock and causal model front-end checks."""
import unittest
import numpy as np
import torch
from bat_internship.bat_audio_dataset import align_waveform, BatAudioDataset
from types import SimpleNamespace


class WaveformTests(unittest.TestCase):
    def test_padding_cropping_and_fractional_grid_rejection(self):
        x = np.arange(1, 81, dtype=float)
        a = align_waveform(x, 1000, [-32, 0, 32], dt_ms=32)
        np.testing.assert_array_equal(a[0, :64], 0)
        np.testing.assert_array_equal(a[0, 64:], x[:32])
        b = align_waveform(x, 1000, [0, 32], audio_offset_ms=-40)
        np.testing.assert_array_equal(b[0], x[8:72])
        with self.assertRaises(ValueError):
            align_waveform(x, 1000, [0, 32], audio_offset_ms=.5)
        with self.assertRaises(ValueError):
            align_waveform(x, 1001, [0, 32])

    def test_native_mel_onset_delay_and_no_future_leakage(self):
        fs = 200000
        x = np.sin(2*np.pi*40000*np.arange(6400)/fs)
        dataset = SimpleNamespace(F=32, dt=32, audio_fs=fs)
        frontend = BatAudioDataset.make_wav2spec(dataset)
        a = align_waveform(x, fs, [-32, 0, 32, 64])
        y = frontend(a[None])
        self.assertEqual(tuple(y.shape), (1,1,32,4))
        self.assertTrue(torch.isfinite(y).all())
        torch.testing.assert_close(y[..., [0,1,3]],
                                  torch.full_like(y[..., [0,1,3]], np.log(1e-4)))
        raw = BatAudioDataset.make_wav2spec(dataset, compression="none")(a[None])
        torch.testing.assert_close(y, raw.clamp_min(1e-4).log())
        self.assertGreater(y[...,2].max(),0)
        delayed = align_waveform(x, fs, [-32,0,32,64], audio_offset_ms=32)
        torch.testing.assert_close(frontend(delayed[None])[...,3], y[...,2])
        altered = a.clone()
        altered[...,3*6400:] = 1
        torch.testing.assert_close(frontend(altered[None])[...,:3],y[...,:3])
