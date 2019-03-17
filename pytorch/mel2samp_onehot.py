# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met: 
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# *****************************************************************************
"""
Generating pairs of mel-spectrograms and original audio
"""
import random
import torch
import torch.utils.data
import numpy as np
import tensorflow as tf

from pprint import pprint

from audio_tf import AudioProcessor
from audio_lws import LwsAudioProcessor

config = tf.ConfigProto(device_count={'GPU': 0})
tf.enable_eager_execution(config=config)

import utils


class Mel2SampOnehot(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, audio_files, mu_quantization, no_chunks, audio_config, segment_length,
                 use_tf=False, use_lws=True, load_mel=False, verbose=False):

        audio_files = utils.files_to_list(audio_files)
        self.audio_files = audio_files
        random.seed(1234)
        random.shuffle(self.audio_files)

        if not load_mel:
            if use_tf:
                audio_processor_cls = AudioProcessor
            elif use_lws:
                audio_processor_cls = LwsAudioProcessor
            else:
                raise ValueError("Mel spectrum can be calculated only with tf or lws!")
            self.audio_processor = audio_processor_cls(audio_config)

        self.mu_quantization = mu_quantization
        self.segment_length = segment_length

        audio_params = AudioProcessor._load_params(audio_config)
        if verbose:
            print("Audio params:")
            pprint(audio_params)
        self.window_length = audio_params["window_size"]
        self.window_step = audio_params["window_step"]
        self.sample_rate = audio_params["sample_rate"]
        self.mel_segment_length = int(np.ceil(
            (segment_length - self.window_length) / self.window_step)
        )
        self.num_mels = audio_params["num_mel_bins"]
        self.use_tf = use_tf
        self.load_mel = load_mel
        self.no_chunks = no_chunks
        self.use_lws = use_lws

    def get_mel(self, audio):
        mel = self.audio_processor.compute_spectrum(audio)
        if self.use_tf:
            mel = mel.numpy()
        return mel

    def __getitem__(self, index):
        # Read audio
        audio_filename, mel_filename = self.audio_files[index]

        audio, sample_rate = utils.load_wav(audio_filename)
        audio /= utils.MAX_WAV_VALUE

        if sample_rate != self.sample_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sample_rate, self.sample_rate))
        if self.no_chunks:
            mel = self.get_mel(audio)
        else:
            if mel_filename != "" and self.load_mel:
                if self.segment_length % self.window_step != 0:
                    raise ValueError("Hop length should be a divider of segment length")
                mel = np.load(mel_filename)
                # Take segment
                if mel.shape[0] >= self.mel_segment_length:
                    max_mel_start = mel.shape[0] - self.mel_segment_length
                    mel_start = random.randint(0, max_mel_start)
                    mel = mel[mel_start: mel_start + self.mel_segment_length]
                    assert mel.shape[0] == self.mel_segment_length
                    audio_start = mel_start * self.window_step
                    audio = audio[audio_start: audio_start + self.segment_length]
                    assert audio.shape[0] == self.segment_length
                else:
                    audio = np.pad(audio, (0, self.segment_length - audio.shape[0]), 'constant')
                    mel = np.pad(mel, (0, 0, 0, self.mel_segment_length - mel.shape[0]), 'constant')
            else:
                if audio.shape[0] >= self.segment_length:
                    max_audio_start = audio.shape[0] - self.segment_length
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[audio_start:audio_start + self.segment_length]
                else:
                    audio = np.pad(audio, (0, self.segment_length - audio.shape[0]), 'constant')
                mel = self.get_mel(audio)

        mel = torch.FloatTensor(mel)
        audio = torch.FloatTensor(audio)
        audio = utils.mu_law_encode(audio, self.mu_quantization)
        return mel, audio
    
    def __len__(self):
        return len(self.audio_files)
