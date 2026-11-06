import unittest
import numpy as np
import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from bat_internship.training import split_stimuli, per_roi_scores, build_model, comparison_scores, training_response_statistics
from deepSTRF.models.wav2spec import make_wav2spec
from bat_internship.training import CheckpointPath


class TrainingTests(unittest.TestCase):
    def test_checkpoint_retries_transient_lock_and_preserves_contents(self):
        with TemporaryDirectory() as tmp:
            source = CheckpointPath(tmp) / 'state.pt.tmp'
            target = Path(tmp) / 'state.pt'
            source.write_bytes(b'new checkpoint')
            target.write_bytes(b'old checkpoint')
            original = type(Path()).replace
            calls = []
            def flaky(path, destination):
                calls.append(path)
                if len(calls) == 1:
                    raise PermissionError('simulated transient lock')
                return original(path, destination)
            with patch.object(type(Path()), 'replace', flaky), patch('bat_internship.training.time.sleep'):
                source.replace(target)
            self.assertEqual(target.read_bytes(), b'new checkpoint')
            self.assertFalse(source.exists())
            source.write_bytes(b'next checkpoint')
            with patch.object(type(Path()), 'replace', side_effect=PermissionError), patch('bat_internship.training.time.sleep'):
                with self.assertRaises(PermissionError):
                    source.replace(target)
            self.assertEqual(target.read_bytes(), b'new checkpoint')
            self.assertTrue(source.exists())

    def test_split_is_complete_disjoint_and_reproducible(self):
        identities = [f"sound-{i}" for i in range(17)]
        split = split_stimuli(identities)
        self.assertEqual(split, split_stimuli(identities))
        self.assertEqual([len(v) for v in split.values()], [11,3,3])
        all_ids = sum(split.values(), [])
        self.assertEqual(sorted(all_ids),list(range(17)))
        with self.assertRaises(ValueError):
            split_stimuli(["same"]*17)

    def test_metrics_perfect_wrong_and_constant(self):
        target = np.arange(24,dtype=float).reshape(2,3,4)
        exact=per_roi_scores(target,target)
        self.assertEqual(exact['mean_mse'],0)
        np.testing.assert_allclose(exact['correlation_per_roi'],1)
        wrong=per_roi_scores(target+2,target)
        self.assertEqual(wrong['mean_mse'],4)
        const=per_roi_scores(np.ones_like(target),target)
        self.assertEqual(const['correlation_per_roi'],[None]*3)

    def test_category_split_preserves_old_test_and_training_coverage(self):
        identities = [f'sound-{i}' for i in range(17)]
        categories = ['Social']*6 + ['Navigation']*6 + ['Noise'] + ['Mouse']*4
        split = split_stimuli(identities, categories=categories)
        self.assertEqual(split, split_stimuli(identities, categories=categories))
        self.assertEqual(split['test'], split_stimuli(identities)['test'])
        self.assertEqual(sorted(sum(split.values(), [])), list(range(17)))
        self.assertEqual([len(v) for v in split.values()], [11,3,3])
        for subset in ['train', 'validation']:
            self.assertEqual({categories[i] for i in split[subset]},
                             {'Social','Navigation','Mouse'})
        with self.assertRaises(ValueError):
            split_stimuli(identities, categories=['Social'])
        single = split_stimuli(identities, categories=['Social']*17)
        self.assertEqual([len(v) for v in single.values()], [11,3,3])

    def test_float32_constants_and_baseline_comparison(self):
        target = np.arange(306,dtype=np.float32).reshape(3,2,51)
        baseline = np.broadcast_to(np.array([0.7,1.1],dtype=np.float32)[None,:,None],
                                   target.shape)
        result = comparison_scores(target, target, baseline)
        self.assertEqual(result['constant_baseline']['correlation_per_roi'],[None,None])
        self.assertEqual(result['relative_mse_improvement'],1)
        self.assertEqual(result['rois_beating_baseline'],2)
        self.assertEqual(result['median_roi_correlation'],1)
        zero = np.zeros_like(target)
        self.assertIsNone(comparison_scores(zero,zero,zero)['relative_mse_improvement'])

    def test_missing_targets_excluded_and_unobserved_roi_is_null(self):
        target = np.array([[[1., 3.], [np.nan, np.nan], [np.nan, np.nan]],
                           [[np.nan, np.nan], [2., 4.], [np.nan, np.nan]]])
        prediction = np.nan_to_num(target) + 2
        scores = per_roi_scores(prediction, target)
        self.assertEqual(scores['mean_mse'], 4)
        self.assertEqual(scores['mse_per_roi'], [4., 4., None])
        self.assertEqual(scores['observed_bins_per_roi'], [2, 2, 0])
        self.assertEqual(scores['correlation_per_roi'], [1., 1., None])
        result = comparison_scores(prediction, target, np.zeros_like(target))
        self.assertEqual(result['rois_beating_baseline'], 2)
        import json
        json.dumps(result, allow_nan=False)
        prediction[0, 2, 0] = np.nan
        with self.assertRaises(ValueError):
            per_roi_scores(prediction, target)
        with self.assertRaises(ValueError):
            per_roi_scores(np.zeros_like(target), np.full_like(target, np.nan))

    def test_training_statistics_missing_sounds_unequal_repeats(self):
        class Dataset:
            N_neurons = 2
            responses = [
                [torch.tensor([[1., 3.], [3., 5.]], dtype=torch.float64), torch.full((1, 1), float('nan'))],
                [torch.full((1, 1), float('nan')), torch.tensor([[6., 8.]], dtype=torch.float64)],
                [torch.full((1, 1), 999.), torch.full((1, 1), 999.)],
            ]
        location, scale, count = training_response_statistics(Dataset(), [0, 1])
        np.testing.assert_allclose(location, [3., 7.])
        np.testing.assert_allclose(scale, [1., 1.])
        np.testing.assert_array_equal(count, [2, 2])
        with self.assertRaises(ValueError):
            training_response_statistics(Dataset(), [0])

    def test_linear_only(self):
        with self.assertRaises(ValueError):
            build_model('convnet2d', type('Dataset', (), dict(F=32, N_neurons=2))())

    def test_native_models_waveform_gradient_and_checkpoint(self):
        class Dataset:
            F=32
            N_neurons=3
            def make_wav2spec(self):
                return make_wav2spec('mel',audio_fs=8000,dt_ms=32,win_ms=32,
                                    n_mels=32,f_min=0,f_max=4000,
                                    magnitude='power',compression='log1p')
        torch.set_num_threads(2)
        x=torch.randn(2,1,51*256)
        for name in ['linear']:
            model=build_model(name,Dataset())
            out=model(x)
            self.assertEqual(tuple(out.shape),(2,3,1,51))
            out.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            restored=build_model(name,Dataset())
            restored.load_state_dict(model.state_dict())
            model.eval(); restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(x),restored(x))
