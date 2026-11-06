"""Portable multi-site real-audio dataset on the assumed calcium time grid.

Inputs: grid-locked float32 waveforms; native deepSTRF wav2spec lives in the model.
Targets: original float64 fluorescence, without baseline subtraction.
Timing and WAV onset remain explicit working assumptions, not confirmations.
"""
import numpy as np
import torch

from numbers import Integral
from .audio_features_legacy import aligned_spectrogram  # historical public helper
from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from .bat_data import BatData
from .audio_mapping import read_audio_manifest, load_mapped_audio


def align_waveform(audio, sample_rate, time_ms, *, dt_ms=32.0, audio_offset_ms=0.0):
    """Grid-lock source audio to bins [time-dt, time), preserving onset and silence.

    Bin 0 starts at time_ms[0]-dt_ms. Source WAV sample 0 occurs at
    audio_offset_ms. Audio outside the neural observation interval is cropped;
    missing context is zero-padded. Return mono float32 (1, T_neural * hop).
    """
    audio = np.asarray(audio)
    times = np.asarray(time_ms, dtype=float)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Expected finite nonempty mono audio")
    if (not np.isfinite([sample_rate, dt_ms, audio_offset_ms]).all()
            or sample_rate <= 0 or dt_ms <= 0 or times.ndim != 1
            or not times.size or not np.isfinite(times).all()
            or (len(times) > 1 and not np.allclose(np.diff(times), dt_ms))):
        raise ValueError("Invalid audio/calcium time grid")
    hop_float = sample_rate * dt_ms / 1000
    start_float = (audio_offset_ms - (times[0] - dt_ms)) * sample_rate / 1000
    if (hop_float < 1 or not np.isclose(hop_float, round(hop_float), rtol=0, atol=1e-6)
            or not np.isclose(start_float, round(start_float), rtol=0, atol=1e-6)):
        raise ValueError("Audio grid must align to integer samples")
    hop, start = int(round(hop_float)), int(round(start_float))
    aligned = np.zeros(len(times) * hop, dtype=np.float32)
    lo, hi = max(0, start), min(len(aligned), start + len(audio))
    if hi > lo:
        aligned[lo:hi] = audio[lo - start:hi - start]
    return torch.from_numpy(aligned[None])


class BatAudioDataset(AudioNeuralDataset):
    """Regular VocalizationMat data with explicit files and audited identities.

    Required paths: path (MAT), audio_root (directory), mapping_path (CSV).
    Use site_indices=[0, 13], or site_index=24 for a single site. Site and ROI
    indices are zero-based. Common sounds merge by verified (sample rate, PCM
    hash), never by category or local index. Conflicting categories and duplicate
    identities within one site fail rather than overwrite recordings.

    Calcium frames must have identical relative grids across selected sites.
    Repetition counts may differ; repeats remain local to each recording site.
    Missing stimulus/ROI pairs use independent (1,1) NaN sentinels. Partial
    nonfinite recordings fail because deepSTRF treats them as missing pairs.
    """

    def __init__(self, path, site_index=None, *, site_indices=None, audio_root,
                 mapping_path, dt_ms=32.0, audio_offset_ms=0.0,
                 return_waveform=True, n_mels=32):
        if site_index is not None and site_indices is not None:
            raise ValueError("Use site_index OR site_indices")
        indices = [site_index] if site_index is not None else list(site_indices or [])
        if (not indices or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                               or i < 0 for i in indices) or len(set(indices)) != len(indices)):
            raise ValueError("Provide distinct nonnegative integer site indices")
        if not np.isfinite([dt_ms, audio_offset_ms]).all() or dt_ms <= 0:
            raise ValueError("Invalid timing assumptions")
        if type(return_waveform) is not bool:
            raise ValueError("return_waveform must be a bool")
        if isinstance(n_mels, bool) or not isinstance(n_mels, Integral) or n_mels < 1:
            raise ValueError("n_mels must be a positive integer")
        super().__init__(str(path), float(dt_ms))
        self.return_waveform = return_waveform
        self.F = int(n_mels)
        self.site_indices = [int(i) for i in indices]
        self.audio_offset_ms = float(audio_offset_ms)
        self.is_placeholder = False
        self.timing_confirmed = False
        self.stimulus_mapping_status = "audited_regular_wav"
        self.site_metadata = []
        self.roi_ranges = []
        self.local_to_global = {}
        self.waveforms = []

        self.common_ids = []
        mappings = read_audio_manifest(mapping_path)
        identities = {}
        records = []
        with BatData(path) as source:
            for site in self.site_indices:
                meta = source.vocal_metadata(site)
                shape = meta["shape_time_repeat_stimulus_roi"]
                frames = np.arange(shape[0]) - meta["onset_frame_index"]
                times = frames * dt_ms
                if self.site_metadata and not np.array_equal(times, self.time_ms):
                    raise ValueError("Selected sites have incompatible relative calcium time grids")
                self.time_ms = times
                self.site_metadata.append(meta)
                expected = set(range(shape[2]))
                actual = {stim for site_id, stim in mappings if site_id == site}
                if actual != expected:
                    raise ValueError(f"Site {site}: manifest must cover exactly all regular stimuli")
                bounds = slice(self.N_neurons, self.N_neurons + shape[3])
                self.roi_ranges.append(bounds)
                for roi in range(shape[3]):
                    self.nrn_meta.append(dict(
                        cell_id=f"bat_{meta['bat_id']}_site_{meta['site_number']}_roi_{roi}",
                        animal_id=meta["bat_id"], site_index=site,
                        site_number=meta["site_number"], roi_index=roi,
                        recording_modality="calcium_imaging",
                        signal="stored_fluorescence_no_conversion",
                        repeat_scope="local_to_site_not_simultaneous_across_sites"))
                self.N_neurons += shape[3]
                self.local_to_global[site] = []
                seen = set()
                for local_s, category in enumerate(meta["categories"]):
                    row = mappings[(site, local_s)]
                    audio, rate = load_mapped_audio(row, audio_root, category)
                    identity = f"pcm_sha256:{row['pcm_sha256']}:sr:{rate}:mono:int16"
                    if identity in seen:
                        raise ValueError(f"Site {site}: duplicate audio identity; trial handling must be explicit")
                    seen.add(identity)
                    origin = dict(site_index=site, local_stimulus_index=local_s,
                                  wav_path=row["wav_path"], repeat_count=shape[1])
                    if identity not in identities:
                        if self.waveforms and rate != self.audio_fs:
                            raise ValueError("Inconsistent audio sample rates")
                        self.audio_fs = int(rate)
                        aligned = align_waveform(audio, rate, times, dt_ms=dt_ms,
                                                 audio_offset_ms=audio_offset_ms)
                        global_s = len(self.waveforms)
                        identities[identity] = global_s
                        self.common_ids.append(identity)
                        self.waveforms.append(audio)
                        self.stims.append(aligned)
                        self.stim_meta.append(dict(
                            name=identity, stimulus_identity=identity, category=category,
                            is_placeholder=False, mapping_status=self.stimulus_mapping_status,
                            timing_confirmed=False, wav_path=row["wav_path"],
                            pcm_sha256=row["pcm_sha256"], sample_rate_hz=rate,
                            audio_duration_ms=len(audio) / rate * 1000,
                            time_ms_assumed=times.tolist(), dt_ms_assumed=float(dt_ms),
                            audio_offset_ms_assumed=float(audio_offset_ms),
                            temporal_bin="[calcium_time - dt_ms, calcium_time)",
                            transform="waveform grid alignment; model owns wav2spec",
                            audio_root=str(audio_root),
                            audio_storage="float32_aligned_waveform_and_original_waveform",
                            response_transform="none; original float64 fluorescence", sources=[origin]))
                    else:
                        global_s = identities[identity]
                        if self.stim_meta[global_s]["category"] != category:
                            raise ValueError("Conflicting categories for a shared audio identity")
                        self.stim_meta[global_s]["sources"].append(origin)
                    self.local_to_global[site].append(global_s)
                    records.append((site, local_s, global_s, bounds))
            for _ in self.waveforms:
                self.responses.append([torch.full((1, 1), float("nan"), dtype=torch.float64)
                                       for _ in range(self.N_neurons)])
            for site, local_s, global_s, bounds in records:
                values = source.dataset(site, "VocalizationMat")[:, :, local_s, :]
                if values.dtype != np.float64 or not np.isfinite(values).all():
                    raise ValueError(f"Site {site}, stimulus {local_s}: expected finite original float64 responses")
                self.responses[global_s][bounds] = list(torch.from_numpy(values).permute(2, 1, 0).unbind(0))
        if not self.return_waveform:
            # Optional native spectrogram branch for conventional spec-input models.
            frontend = self.make_wav2spec()
            with torch.no_grad():
                self.stims = [frontend(w.unsqueeze(0)).squeeze(0) for w in self.stims]
            for meta in self.stim_meta:
                meta["transform"] = "native deepSTRF CausalMelSpectrogram"
                meta["audio_storage"] = "precomputed_spectrogram_and_original_waveform"
        self.validate()


    def make_wav2spec(self, **kwargs):
        """Build the official model front-end with matching audio rate and hop.

        Default: 32-ms window/hop, power, log1p, 0..Nyquist, HTK Mel bands.
        These are explicit starting parameters, not a bat hearing model.
        """
        from deepSTRF.models.wav2spec import make_wav2spec
        settings = dict(n_mels=self.F, win_ms=self.dt, f_min=0.0,
                        f_max=self.audio_fs / 2, magnitude="power", compression="log1p")
        settings.update(kwargs)
        if settings["n_mels"] != self.F:
            raise ValueError("Front-end n_mels must match dataset.F")
        return make_wav2spec("mel", audio_fs=self.audio_fs, dt_ms=self.dt, **settings)

    def _concat_check_compat(self, other):
        super()._concat_check_compat(other)
        if self.return_waveform != getattr(other, "return_waveform", False):
            raise ValueError("Cannot concatenate waveform and spectrogram modes")

    def _concat_copy_attrs(self, source):
        super()._concat_copy_attrs(source)
        self.return_waveform = source.return_waveform
        self.is_placeholder = False
        self.timing_confirmed = False
        self.stimulus_mapping_status = source.stimulus_mapping_status



def plot_audio_example(dataset, stimulus_index=10, roi_index=3):
    """Show the original waveform, detailed WAV spectrum and calcium targets."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    if not 0 <= stimulus_index < len(dataset.stims) or not 0 <= roi_index < dataset.N_neurons:
        raise ValueError("Stimulus or ROI index out of bounds")
    meta = dataset.stim_meta[stimulus_index]
    if not torch.isfinite(dataset.responses[stimulus_index][roi_index]).any():
        raise ValueError("This ROI was not measured for the selected stimulus")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    offset_ms = meta["audio_offset_ms_assumed"]
    if hasattr(dataset, "waveforms"):
        waveform = dataset.waveforms[stimulus_index]
    else:
        row = dict(meta, match_status="exact", matlab_category=meta["category"])
        waveform, _ = load_mapped_audio(row, meta["audio_root"], meta["category"])
    wav_time = np.arange(len(waveform)) * 1000 / dataset.audio_fs + offset_ms
    axes[0].plot(wav_time, waveform, lw=0.6, color="#365c7a")
    axes[0].set(ylabel="Audio (PCM / 32768)", title="Original-WAV, ohne zeitliche Streckung")
    # Fine-time display only: do not replace or relabel the 32-ms model features.
    nfft = min(256, len(waveform))
    spectrum_colors = LinearSegmentedColormap.from_list(
        "blue_cyan_yellow", ["#352a87", "#1464ef", "#06b7d7", "#42bd71", "#bddf26", "#fff126"])
    power, _, _, spectrum = axes[1].specgram(
        waveform, NFFT=nfft, Fs=dataset.audio_fs, noverlap=3 * nfft // 4,
        window=np.hanning(nfft), detrend="none", mode="psd", scale="dB",
        cmap=spectrum_colors)
    peak_db = 10 * np.log10(max(float(np.max(power)), np.finfo(float).tiny))
    spectrum.set_clim(peak_db - 60, peak_db)
    x0, x1, y0, y1 = spectrum.get_extent()
    spectrum.set_extent((x0 * 1000 + offset_ms,
                         x1 * 1000 + offset_ms, y0 / 1000, y1 / 1000))
    duration_ms = len(waveform) / dataset.audio_fs * 1000
    axes[1].set(ylabel="Frequency (kHz)", xlabel="Time (ms from assumed sound onset)",
                ylim=(0, dataset.audio_fs / 2000),
                title=f"WAV spectrogram: {nfft / dataset.audio_fs * 1000:g}-ms Hann window, 75% overlap; display only")
    fig.colorbar(spectrum, ax=axes[1], label="PSD (dB re 1 PCM²/Hz)", pad=0.02)
    for ax in axes[:2]:
        ax.set_xlim(offset_ms, offset_ms + duration_ms)
    times = np.asarray(meta["time_ms_assumed"])
    dt = meta["dt_ms_assumed"]
    responses = dataset.responses[stimulus_index][roi_index].numpy()
    for i, response in enumerate(responses):
        axes[2].plot(times, response, color="#7898ae", alpha=0.45, lw=0.8,
                     label="Einzelwiederholungen" if i == 0 else None)
    axes[2].plot(times, responses.mean(axis=0), color="#192f40", lw=2, label="Mittelwert (nur Anzeige)")
    local_roi = dataset.nrn_meta[roi_index]["roi_index"]
    axes[2].set(ylabel="Gespeicherte Fluoreszenz", title=f"Zielwerte: lokale ROI {local_roi} (global {roi_index}), {len(responses)} Wiederholungen unverändert",
                xlabel=f"Zeit (ms); Annahmen: {dt:g} ms/Frame, WAV-Start bei {offset_ms:g} ms")
    axes[2].legend(frameon=False)
    for ax in axes[2:]:
        ax.axvline(0, ls="--", color="0.5", lw=0.8)
        ax.set_xlim(times[0] - dt, times[-1])
    fig.suptitle(f"Site-Index {dataset.nrn_meta[roi_index]['site_index']} · globaler Stimulus {stimulus_index} ({meta['category']}) · echte Audiodaten")
    return fig
