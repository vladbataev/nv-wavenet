import json
import librosa
import math
import numpy as np


import tensorflow as tf


LOG10_TO_LN = math.log(10)
LN_TO_LOG10 = 1 / LOG10_TO_LN
DB_TO_LN = LOG10_TO_LN / 20
LN_TO_DB = 20 * LN_TO_LOG10  # 20 as the power is proportional to power of amplitude


class AudioProcessor:
    def __init__(self, audio_config):
        params = self._load_params(audio_config)
        for k, v in params.items():
            self.__setattr__(k, v)

    @staticmethod
    def _load_params(filepath):
        with open(filepath) as fin:
            params = json.load(fin)
        return params

    def _name_to_window_fn(self, name):
        mapping = {
            "hann": tf.contrib.signal.hann_window,
            "hamming": tf.contrib.signal.hamming_window,
        }
        return mapping[name]

    def preemphasis(self, signals):
        paddings = [
            [0, 0],
            [0, 0],
            [1, 0]
        ]
        emphasized = tf.pad(signals[:, :, :-1], paddings=paddings) * -self.preemphasis_coef + signals
        return emphasized

    def amp_to_db(self, signal):
        return LN_TO_DB * tf.log(tf.maximum(self.min_level, signal))

    def dbfs_normalize(self, signal):
        max_value = tf.reduce_max(signal, axis=[1, 2, 3], keepdims=True)
        return signal - max_value

    def normalize_and_clip_db(self, signal_db):
        """
        Clips signal in decibels to [0; -min_level_db] and then normalizes it to [-max_abs_value; max_abs_value]
        in case symmetric output or to [0; max_abs_value] otherwise.
        :param signal_db:
        :return: clipped signal in decibels to [-max_abs_value; max_abs_value] or [0; max_abs_value].
        """
        clipped = signal_db - self.min_level_db
        normalized = tf.clip_by_value(clipped / -self.min_level_db, 0, 1)
        if self.symmetric_output:
            normalized = (normalized * 2 - 1)
            # so output now in [-1; 1]
        normalized *= self.max_abs_value
        return normalized

    def linear_scale_to_normalized_log_scale(self, spectrogram):
        spectrogram_db = self.amp_to_db(spectrogram)
        if self.dbfs_normalization:
            spectrogram_db = self.dbfs_normalize(spectrogram_db)
        spectrogram_db += self.ref_level_db
        return self.normalize_and_clip_db(spectrogram_db)

    def _mel_basis(self):
        if self.use_tf_mel_basis:
            mel_basis = tf.contrib.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, self.window_size // 2 + 1, self.sample_rate,
                self.lower_edge_hertz, self.upper_edge_hertz
            )
        else:
            mel_basis = librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.window_size,
                n_mels=self.num_mel_bins,
                fmin=self.lower_edge_hertz,
                fmax=self.upper_edge_hertz
            )
            mel_basis = tf.convert_to_tensor(np.transpose(mel_basis, (1, 0)), dtype=tf.float32)
        return mel_basis

    def compute_spectrum(self, signal):
        """
        :param signals: shape [batch_size, 1, num_timestamps]
        :param lengths:
        :param sample_rate:
        :return:
        """
        with tf.name_scope("extract_feats"):
            frame_length = self.window_size
            frame_step = self.window_step
            signals = signal[None, None, ...]
            if self.apply_preemphasis:
                signals = self.preemphasis(signals)
            stfts = tf.contrib.signal.stft(signals, frame_length=frame_length, frame_step=frame_step,
                                           fft_length=frame_length,
                                           window_fn=self._name_to_window_fn(self.window_fn_name),
                                           pad_end=True)
            linear_spectrograms = tf.abs(stfts)
            mel_spectrograms = tf.tensordot(linear_spectrograms, self._mel_basis(), 1)
            normed_mel_spectrograms_db = self.linear_scale_to_normalized_log_scale(mel_spectrograms)
        return tf.transpose(normed_mel_spectrograms_db[0, 0], (1, 0))
