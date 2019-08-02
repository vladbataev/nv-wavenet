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
import os
import torch
import numpy as np
import nv_wavenet
import utils
import json
import librosa
import scipy

from tensorboardX import SummaryWriter
from mel2samp_onehot import Mel2SampOnehot
from audio_tf import AudioProcessor

import tensorflow as tf

tf.enable_eager_execution()


def chunker(seq, size):
    """
    https://stackoverflow.com/a/434328
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def normalize_audio(signal):
    max_value = np.max(signal)
    min_value = np.min(signal)
    return 2 * (signal - min_value) / (max_value - min_value) - 1


def get_taco_output_path(manifest_path):
    with open(manifest_path) as fin:
        line = fin.readline().strip()
        return "/".join(line.split("/")[:-2])


def get_latest_checkpoint(model_dir):
    files = os.listdir(model_dir)
    latest_checkpoint = ""
    last = 0
    for f in files:
        if "wavenet_" in f:
            number = int(f.split("_")[1])
            if number > last:
                last = number
                latest_checkpoint = f
    return os.path.join(model_dir, latest_checkpoint)


def main(input_files, model_dir, output_dir, batch_size, implementation, data_config, audio_config, preload_mels=False):
    model_filename = get_latest_checkpoint(model_dir)
    print("Model path: {}".format(model_filename))
    model = torch.load(model_filename)['model']
    wavenet = nv_wavenet.NVWaveNet(**(model.export_weights()))
    print("Wavenet num layers: {}, max_dilation: {}".format(wavenet.num_layers, wavenet.max_dilation))
    writer = SummaryWriter(output_dir)
    mel_extractor = Mel2SampOnehot(audio_config=audio_config, **data_config)
    input_files = utils.files_to_list(input_files)

    audio_processor = AudioProcessor(audio_config)
    for j, files in enumerate(chunker(input_files, batch_size)):
        mels = []
        for i, file_path in enumerate(files):
            if preload_mels:
                mel = np.load(file_path[0]).T
                mel = torch.from_numpy(mel)
                mel = utils.to_gpu(mel)
            else:
                audio, _ = utils.load_wav_to_torch(file_path)
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                writer.add_audio("eval_true/{}/{}".format(i, file_name), audio / utils.MAX_WAV_VALUE, 0, 22050)
                mel = mel_extractor.get_mel(audio)
                mel = mel.t().cuda()
            mels.append(torch.unsqueeze(mel, 0))
        mels = torch.cat(mels, 0)
        cond_input = model.get_cond_input(mels)
        audio_data = wavenet.infer(cond_input, implementation)

        for i, file_path in enumerate(files):
            file_name = os.path.splitext(os.path.basename(file_path[0]))[0]
            audio = utils.mu_law_decode_numpy(audio_data[i,:].cpu().numpy(), 256)
            print("Range of {}.wav before deemphasis : {} to {}".format(file_name, audio.min(), audio.max()))
            if mel_extractor.apply_preemphasis:
                audio = audio.astype("float32")
                audio = audio_processor.deemphasis(audio[None, :])
                audio = audio.numpy()[0]
            print("Range of {}.wav after deemphasis : {} to {}".format(file_name, audio.min(), audio.max()))
            audio = np.tanh(audio)
            output_filepath = "{}.wav".format(file_name)
            output_filepath = os.path.join(output_dir, output_filepath)
            assert audio.dtype in [np.float64, np.float32]
            assert (np.abs(audio)).max() <= 1
            writer.add_audio(output_filepath, audio, 0, 22050)
            audio = (audio * 32767).astype("int16")
            scipy.io.wavfile.write(output_filepath, 22050, audio)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', "--model_dir", required=True)

    parser.add_argument('-o', "--output_dir", type=str, default=None)
    parser.add_argument('-b', "--batch_size", default=1)
    parser.add_argument('-a', "--audio_config", required=True)
    parser.add_argument('-d', "--data_config", required=True)
    parser.add_argument('-i', "--implementation", type=str, default="persistent",
                        help="""Which implementation of NV-WaveNet to use.
                        Takes values of single, dual, or persistent""" )
    parser.add_argument("--preload_mels", action="store_true")

    args = parser.parse_args()
    if args.implementation == "auto":
        implementation = nv_wavenet.Impl.AUTO
    elif args.implementation == "single":
        implementation = nv_wavenet.Impl.SINGLE_BLOCK
    elif args.implementation == "dual":
        implementation = nv_wavenet.Impl.DUAL_BLOCK
    elif args.implementation == "persistent":
        implementation = nv_wavenet.Impl.PERSISTENT
    else:
        raise ValueError("implementation must be one of auto, single, dual, or persistent")

    with open(args.data_config) as f:
        data_config = json.loads(f.read())

    main(args.filelist_path, args.model_dir, args.output_dir, args.batch_size,
         implementation, data_config, args.audio_config, args.preload_mels)
