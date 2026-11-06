"""Portable synthetic two-site tests: reordered stimuli and unequal repeats."""
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np
import torch
from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.utils.data import neural_collate
from bat_internship.bat_audio_dataset import BatAudioDataset


class Source:
    def __init__(self, path):
        self.arrays = {0: np.arange(12, dtype=np.float64).reshape(3, 2, 1, 2),
                       13: np.arange(18, dtype=np.float64).reshape(3, 3, 2, 1) + 100}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def vocal_metadata(self, site):
        return dict(site_index=site, site_number=site + 1, bat_id=1,
                    shape_time_repeat_stimulus_roi=list(self.arrays[site].shape),
                    onset_frame_index=1, categories=["Social"] * self.arrays[site].shape[2])
    def dataset(self, site, field):
        return self.arrays[site]


class MultiSiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for site, stim, sound in [(0, 0, 1), (13, 0, 2), (13, 1, 1)]:
            pcm = (np.arange(400, dtype=np.int16) * sound).astype('<i2').tobytes()
            name = f"sound{sound}.wav"
            with wave.open(str(self.root / name), 'wb') as wav:
                wav.setparams((1, 2, 200000, 400, 'NONE', 'not compressed'))
                wav.writeframes(pcm)
            self.rows.append(dict(matrix='VocalizationMat', site_index_python=site,
                                  index_python=stim, wav_path=name, match_status='exact',
                                  matlab_category='Social', sample_rate_hz=200000,
                                  pcm_sha256=hashlib.sha256(pcm).hexdigest()))
        self.manifest = self.root / 'mapping.csv'
        self.write_manifest()
    def write_manifest(self):
        with self.manifest.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
    def build(self, source=Source, **kwargs):
        with patch('bat_internship.bat_audio_dataset.BatData', source):
            return BatAudioDataset('arbitrary.mat', audio_root=self.root,
                                   mapping_path=self.manifest, **kwargs)
    def test_direct_base_reordered_shared_audio_and_preserved_values(self):
        self.assertEqual(BatAudioDataset.__bases__, (AudioNeuralDataset,))
        ds = self.build(site_indices=[0, 13])
        self.assertEqual(ds.local_to_global, {0: [0], 13: [1, 0]})
        self.assertEqual((len(ds), ds.N_neurons), (2, 3))
        for site, bounds in zip(ds.site_indices, ds.roi_ranges):
            for local, global_s in enumerate(ds.local_to_global[site]):
                actual = torch.stack(ds.responses[global_s][bounds]).numpy().transpose(2, 1, 0)
                np.testing.assert_array_equal(actual, Source('').arrays[site][:, :, local, :])
        self.assertEqual(len(ds.stim_meta[0]['sources']), 2)
        self.assertEqual(ds.nrn_meta[2]['site_index'], 13)
    def test_missing_pairs_and_unequal_repeats_in_real_collate(self):
        ds = self.build(site_indices=[0, 13])
        b = neural_collate([ds[0], ds[1]])
        self.assertEqual(tuple(b['responses'].shape), (2, 3, 3, 3))
        self.assertFalse(b['valid_mask'][1, :2].any())
        self.assertFalse(b['valid_mask'][0, :2, 2].any())
        self.assertTrue(b['valid_mask'][0, :2, :2].all())
        self.assertTrue(b['valid_mask'][:, 2].all())
        self.assertTrue(torch.equal(b['valid_mask'], torch.isfinite(b['responses'])))
    def test_single_site_and_reversed_site_order(self):
        ds = self.build(site_index=13)
        self.assertEqual(ds.local_to_global[13], [0, 1])
        rev = self.build(site_indices=[13, 0])
        self.assertEqual(rev.local_to_global, {13: [0, 1], 0: [1]})
    def test_native_waveform_grid_and_model_frontend(self):
        ds = self.build(site_indices=[0, 13])
        self.assertIsInstance(ds.stims, list)
        self.assertTrue(ds.return_waveform)
        self.assertEqual(ds.hop, 6400)
        self.assertEqual(tuple(ds.stims[0].shape), (1, 19200))
        self.assertEqual(ds.stims[0].dtype, torch.float32)
        # Neural times [-32, 0, 32]: WAV starts in bin 2, not bin 1.
        self.assertEqual(torch.count_nonzero(ds.stims[0][:, :12800]), 0)
        np.testing.assert_array_equal(ds.stims[0][0,12800:13200],
                                      ds.waveforms[0].astype(np.float32))
        ds.validate()
        frontend = ds.make_wav2spec()
        batch = neural_collate([ds[0]])
        self.assertEqual(tuple(frontend(batch['stims']).shape), (1,1,32,3))
        self.assertEqual(batch['responses'].dtype, torch.float32)
        self.assertEqual(ds.responses[0][0].dtype, torch.float64)
        ds.select_stim(1)
        torch.testing.assert_close(ds[0]['stims'],ds.stims[1])

    def test_native_standardization_and_concatenation(self):
        from deepSTRF.utils.data import concat_neural_datasets
        a = self.build(site_index=0)
        b = self.build(site_index=13)
        raw = a.waveforms[0].copy()
        stats = a.standardize_stims(stim_indices=[0])
        self.assertTrue(torch.isfinite(a.stims[0]).all())
        np.testing.assert_array_equal(a.waveforms[0], raw)
        # Rebuild for concatenation so the inputs share preprocessing.
        a = self.build(site_index=0)
        for ds in [a + b, concat_neural_datasets([a,b])]:
            ds.validate()
            self.assertTrue(ds.return_waveform)
            self.assertEqual(len(ds.stims),3)
            self.assertEqual(ds.N_neurons,3)
            batch=neural_collate([ds[0],ds[1]])
            self.assertFalse(batch['valid_mask'][0,2].any())
            self.assertEqual(tuple(ds.make_wav2spec()(batch['stims']).shape),(2,1,32,3))
        spec = self.build(site_index=0, return_waveform=False)
        self.assertEqual(tuple(spec.stims[0].shape),(1,32,3))
        torch.testing.assert_close(spec.stims[0],a.make_wav2spec()(a.stims[0][None])[0])
        with self.assertRaises(ValueError):
            _ = a + spec

    def test_bad_selection(self):
        for kwargs in [dict(site_indices=[]), dict(site_indices=[0, 0]),
                       dict(site_index=True), dict(site_index=0, site_indices=[13])]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)
    def test_missing_duplicate_or_unconfirmed_mapping(self):
        original = [r.copy() for r in self.rows]
        for rows in [original[:-1], original + [original[0]],
                     [{**r, 'match_status': 'candidate'} for r in original]]:
            self.rows = rows
            self.write_manifest()
            with self.assertRaises(ValueError):
                self.build(site_indices=[0, 13])
    def test_changed_audio(self):
        self.rows[2]['pcm_sha256'] = 'wrong'
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.build(site_indices=[0, 13])
    def test_conflicting_category_and_duplicate_identity(self):
        class OtherCategory(Source):
            def vocal_metadata(self, site):
                m = super().vocal_metadata(site)
                if site == 13:
                    m['categories'][1] = 'Navigation'
                return m
        self.rows[2]['matlab_category'] = 'Navigation'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Conflicting categories'):
            self.build(source=OtherCategory, site_indices=[0, 13])
        self.rows[2] = {**self.rows[0], 'site_index_python': 13, 'index_python': 1}
        self.rows[1] = {**self.rows[0], 'site_index_python': 13, 'index_python': 0}
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'duplicate audio identity'):
            self.build(site_indices=[0, 13])
    def test_time_mismatch_and_partial_nan_rejected(self):
        class BadTime(Source):
            def vocal_metadata(self, site):
                m = super().vocal_metadata(site)
                if site == 13:
                    m['onset_frame_index'] = 0
                return m
        class BadValues(Source):
            def __init__(self, path):
                super().__init__(path)
                self.arrays[13][0, 0, 0, 0] = np.nan
        for source in [BadTime, BadValues]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.build(source=source, site_indices=[0, 13])


if __name__ == '__main__':
    unittest.main()
