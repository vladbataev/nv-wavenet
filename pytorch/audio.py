import tensorflow as tf
import math
import argparse
import tqdm
import librosa
import numpy as np
import os
import functools

LOG10_TO_LN = math.log(10)
LN_TO_LOG10 = 1 / LOG10_TO_LN
DB_TO_LN = LOG10_TO_LN / 20
LN_TO_DB = 20 * LN_TO_LOG10  # 20 as the power is proportional to power of amplitude


class AudioProcessor:
    def __init__(self, min_level_db=-100.0, window_size=1024, window_step=256, preemphasis_coef=0.97,
                 lower_edge_hertz=75.0, upper_edge_hertz=7600.0, num_mel_bins=80, ref_level_db=-20.0, post_power=1.0,
                 window_fn_name="hann", dbfs_normalization=True, apply_preemphasis=True):
        self.window_size = window_size
        self.window_step = window_step
        self.preemphasis_coef = preemphasis_coef
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz
        self.num_mel_bins = num_mel_bins
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db
        self.post_power = post_power
        self._name_to_window_fn = {
            "hann": tf.contrib.signal.hann_window,
            "hamming": tf.contrib.signal.hamming_window,
        }
        self.window_fn = self._name_to_window_fn[window_fn_name]
        self.dbfs_normalization = dbfs_normalization
        self.apply_preemphasis = apply_preemphasis

    def preemphasis(self, signals):
        paddings = [
            [0, 0],
            [0, 0],
            [1, 0]
        ]
        emphasized = tf.pad(signals[:, :, :-1], paddings=paddings) * -self.preemphasis_coef + signals
        return emphasized

    def deemphasis(self, signal):
        fir_approximation = [1]
        for i in range(math.ceil(1 / (1 - self.preemphasis_coef))):
            fir_approximation.append(fir_approximation[-1] * self.preemphasis_coef)
        filters = tf.constant(fir_approximation[::-1], dtype=tf.float32, shape=(len(fir_approximation), 1, 1))
        paddings = [
            [0, 0],
            [len(fir_approximation), 0],
        ]
        signal = tf.pad(signal, paddings)
        return tf.nn.conv1d(signal[:, :, None], filters, 1, data_format="NWC", padding="VALID")[:, :, 0]

    def amp_to_db(self, signal):
        return LN_TO_DB * tf.log(signal)

    def db_to_amp(self, signal):
        return tf.exp(signal * DB_TO_LN)

    def dbfs_normalize(self, signal):
        max_value = tf.reduce_max(signal, axis=[1, 2, 3], keepdims=True)
        return signal - max_value

    def normalize_and_clip_db(self, signal_db):
        """
        Clips signal in decibels to [0; -min_level_db] and then normalizes it to [0; 1].
        :param signal_db:
        :return: clipped signal in decibels to [0; 1]
        """
        clipped = signal_db - self.min_level_db
        return tf.clip_by_value(clipped / -self.min_level_db, 0, 1)

    def linear_scale_to_normalized_log_scale(self, spectrogram):
        spectrogram_db = self.amp_to_db(spectrogram)
        if self.dbfs_normalization:
            spectrogram_db = self.dbfs_normalize(spectrogram_db)
        spectrogram_db += self.ref_level_db
        return self.normalize_and_clip_db(spectrogram_db)

    def normalized_log_scale_to_linear_scale(self, spectrogram):
        spectrogram = spectrogram * (-self.min_level_db) + self.min_level_db
        return self.db_to_amp(spectrogram)

    def reconstruct_audio(self, linear_spectrogram):
        linear_spectrogram = self.normalized_log_scale_to_linear_scale(linear_spectrogram) ** self.post_power
        reconstructed_audio = griffin_lim(linear_spectrogram, window_size=self.window_size,
                                          window_step=self.window_step, window_fn=self.window_fn)
        if self.apply_preemphasis:
            reconstructed_audio = self.deemphasis(reconstructed_audio)
        return reconstructed_audio / tf.reduce_max(tf.abs(reconstructed_audio), axis=-1, keepdims=True)

    def compute_spectrum(self, signals, lengths, sample_rate):
        """
        :param signals: shape [batch_size, 1, num_timestamps]
        :param lengths:
        :param sample_rate:
        :return:
        """
        with tf.name_scope("extract_feats"):
            frame_length = self.window_size
            frame_step = self.window_step
            if self.apply_preemphasis:
                signals = self.preemphasis(signals)
            stfts = tf.contrib.signal.stft(signals, frame_length=frame_length, frame_step=frame_step,
                                           fft_length=frame_length,
                                           window_fn=self.window_fn,
                                           pad_end=True)
            linear_spectrograms = tf.abs(stfts)
            normed_linear_spectrograms_db = self.linear_scale_to_normalized_log_scale(linear_spectrograms)
            num_spectrogram_bins = linear_spectrograms.shape[-1].value
            linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, num_spectrogram_bins, sample_rate, self.lower_edge_hertz, self.upper_edge_hertz)
            mel_spectrograms = tf.tensordot(linear_spectrograms, linear_to_mel_weight_matrix, 1)
            normed_mel_spectrograms_db = self.linear_scale_to_normalized_log_scale(mel_spectrograms)
            lengths = tf.ceil(tf.cast(lengths, dtype=tf.float32) / frame_step)

        return normed_linear_spectrograms_db[:, 0], normed_mel_spectrograms_db[:, 0], lengths


def griffin_lim(spectrogram, n_iter=5, window_size=1024, window_step=256, window_fn=tf.contrib.signal.hann_window):
    with tf.name_scope("griffin_lim"):
        stft_magnitude = tf.cast(spectrogram, tf.complex64)
        signal = tf.contrib.signal.inverse_stft(stft_magnitude, window_size, window_step,
                                                window_fn=functools.partial(window_fn, periodic=True))
        for i in range(n_iter):
            with tf.name_scope("iter_{}".format(i)):
                stft = tf.contrib.signal.stft(signal, window_size, window_step)
                phase = stft / tf.cast(1e-8 + tf.abs(stft), tf.complex64)
                signal = tf.contrib.signal.inverse_stft(stft_magnitude * phase, window_size, window_step,
                                                        window_fn=functools.partial(window_fn, periodic=True))
    return signal


def main():
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_filelist", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--min_level_db", type=float, default=-100.0)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--window_step", type=int, default=256)
    parser.add_argument("--preemphasis_coef", type=float, default=0.97)
    parser.add_argument("--lower_edge_hertz", type=float, default=125.0)
    parser.add_argument("--upper_edge_hertz", type=float, default=7600.0)
    parser.add_argument("--num_mel_bins", type=int, default=80)
    parser.add_argument("--ref_level_db", type=float, default=-20.0)
    parser.add_argument("--post_power", type=float, default=1.5)
    parser.add_argument("--dbfs_normalization", type=bool, default=True)
    parser.add_argument("--apply_preemphasis", type=bool, default=True)
    args = parser.parse_args()

    audio_filelist = args.audio_filelist
    sample_rate = args.sample_rate
    output_dir = args.output_dir
    audio_kwargs = vars(args)
    audio_kwargs.pop("audio_filelist")
    audio_kwargs.pop("sample_rate")
    audio_kwargs.pop("output_dir")
    audio_processor = AudioProcessor(**audio_kwargs)
    with open(audio_filelist) as fin:
        for line in tqdm.tqdm(fin):
            audio_filepath = line.strip()
            signal, sr = librosa.load(audio_filepath, sr=sample_rate)
            lengths = np.array([signal.shape[0]])[None, :]
            signals = signal[None, None,]
            _, mels, _ = audio_processor.compute_spectrum(signals, lengths, sample_rate)
            base_name = os.path.basename(audio_filepath)
            mel_fname = base_name.split(".")[0] + ".npy"
            np.save(os.path.join(output_dir, mel_fname), mels[0])


if __name__ == "__main__":
    main()
