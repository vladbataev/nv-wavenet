import json
import librosa
import lws
import numpy as np


class LwsAudioProcessor:
    def __init__(self, audio_config):
        params = self._load_params(audio_config)
        self._params = params
        self._mel_basis = self._compute_mel_basis()
        self._lws_processor = self._build_lws_processor()

    @staticmethod
    def _load_params(filepath):
        with open(filepath) as fin:
            params = json.load(fin)
        return params

    def _compute_mel_basis(self):
        mel_basis = librosa.filters.mel(
            sr=self._params["sample_rate"],
            n_fft=self._params["window_size"],
            n_mels=self._params["num_mel_bins"],
            fmin=self._params["lower_edge_hertz"],
            fmax=self._params["upper_edge_hertz"],
        )
        return mel_basis

    def _build_lws_processor(self):
        return lws.lws(self._params["window_size"], self._params["window_step"],
                       fftsize=self._params["window_size"], mode="speech")

    def _amp_to_db(self, x, min_level_db):
        min_level = np.exp(min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _stft(self, wav):
        return self._lws_processor.stft(wav).T

    def _linear_to_mel(self, linear):
        return np.dot(self._mel_basis, linear)

    def melspectrogram(self, wav):
        D = self._stft(wav)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D)),
                            self._params["min_level_db"]) - self._params["ref_level_db"]
        mel = self._normalize(S, self._params["max_abs_value"], self._params["min_level_db"])
        # as lws by default pad from left and right (window_size - hop_size) // hop_size
        mel = mel[:, (self._params["window_size"] - self._params["window_step"]) // self._params["window_step"]:]
        return mel

    def _normalize(self, S, max_abs_value, min_level_db):
        return np.clip((2 * max_abs_value) * (
                (S - min_level_db) / (-min_level_db)) - max_abs_value, -max_abs_value, max_abs_value)
